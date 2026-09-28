"""
Drug name -> SMILES via SRI Name Resolution + Node Normalizer + PubChem
--------------------------------------------------------------------------
A library of functions for resolving a chemical/drug name to a SMILES
string via a more robust, synonym-aware path than querying PubChem's name
endpoint directly (see name_to_smiles_via_pubchem_direct.py for that
simpler, more direct approach):

  1. Resolve the name to a standardized curie via the SRI Name Resolution
     service (handles synonyms/abbreviations better than a literal string
     match)
  2. Look up that curie's equivalent identifiers via the SRI Node
     Normalizer
  3. Pick out any PubChem CID among those equivalent identifiers
  4. Query PubChem for that CID's Isomeric SMILES

This file is not wired into the batch pipeline yet -- the __main__ block
below is just a small demo run against 3 example drug names, not the full
dataset. See name_to_smiles_via_pubchem_direct.py for the script that
actually processes pk_metadata_figure_level.csv end-to-end.
"""
import io, re, requests, json
from functools import cache
import pandas as pd
from io import StringIO
from typing import Optional


class IDList:
    """
    A class to manage a list of IDs with a method to check for PubChem IDs.
    """

    def __init__(self, ids=None):
        """
        Initialize the IDList with an optional list of IDs.

        Args:
            ids (list, optional): Initial list of IDs. Defaults to empty list.
        """
        self.ids = ids if ids is not None else []

    def add_id(self, id_value):
        self.ids.append(id_value)

    def remove_id(self, id_value):
        if id_value in self.ids:
            self.ids.remove(id_value)

    def contains_pubchem(self):
        """
        Check if the list contains a PubChem ID and return the first one found.

        PubChem IDs follow the pattern: "PUBCHEM:XXXXXX" where X are digits.

        Returns:
            str or False: The first PubChem ID found, or False if none exists.
        """
        # Pattern to match PubChem IDs (case-insensitive)
        pubchem_pattern = re.compile(r'^PUBCHEM.COMPOUND:\d+$', re.IGNORECASE)

        for id_value in self.ids:
            if isinstance(id_value, str) and pubchem_pattern.match(id_value):
                return id_value

        return False

    def get_all_pubchem_ids(self):
        """
        Get all PubChem IDs in the list.

        Returns:
            list: List of all PubChem IDs found.
        """
        pubchem_pattern = re.compile(r'^PUBCHEM.COMPOUND:\d+$', re.IGNORECASE)
        return [id_value for id_value in self.ids
                if isinstance(id_value, str) and pubchem_pattern.match(id_value)]

    def __len__(self):
        """Return the number of IDs in the list."""
        return len(self.ids)

    def __str__(self):
        """String representation of the IDList."""
        return f"IDList({self.ids})"

    def __repr__(self):
        """Developer representation of the IDList."""
        return f"IDList(ids={self.ids!r})"


def normalizer_alt_ids(id: str) -> IDList:
    """
    Uses Node Normalizer to return a set of IDs
    """
    return IDList([id])


def get_smiles_from_pubchem(pubchem_id: int) -> Optional[str]:
    """
    Retrieve the SMILES string for a chemical compound using its PubChem ID (CID).

    Args:
        pubchem_id (int): The PubChem Compound ID (CID)

    Returns:
        Optional[str]: The SMILES string if found, None if not found or error occurs

    Raises:
        ValueError: If the pubchem_id is not a positive integer
        requests.RequestException: If there's an error with the API request
    """
    if not isinstance(pubchem_id, int) or pubchem_id <= 0:
        raise ValueError("PubChem ID must be a positive integer")

    # PubChem REST API endpoint
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    endpoint = f"{base_url}/compound/cid/{pubchem_id}/property/IsomericSMILES/JSON"
    print(f"Requesting: {endpoint}")

    try:
        # Make the API request
        response = requests.get(endpoint, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Parse the JSON response
        data = response.json()

        # Extract the SMILES string - use IsomericSMILES, not SMILES
        smiles = data["PropertyTable"]["Properties"][0]["IsomericSMILES"]
        return smiles

    except requests.RequestException as e:
        print(f"API request error: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Error parsing PubChem response: {e}")
        return None


@cache
def nameres(itemRequest: str) -> Optional[str]:
    """Resolve chemical name to an identifier using name resolution service"""
    print('Resolving name...')
    failed_counts = 0

    while failed_counts < 5:
        try:
            response = requests.get(itemRequest, timeout=10)
            response.raise_for_status()
            returned = pd.read_json(StringIO(response.text))
            return returned.curie[0]
        except Exception as e:
            print(f"Attempt {failed_counts + 1} failed: {e}")
            failed_counts += 1

    raise Exception(f"Failed to resolve name after 5 attempts for: {itemRequest}")


def identify(name: str, params: dict) -> str:
    """
    Identify a chemical name and return its identifier.

    Args:
        name (str): string to be identified
        params (dict): name resolver parameters to feed into get request

    Returns:
        str: ID most closely matching string.
    """
    itemRequest = (params['url'] +
                   params['service'] +
                   '?string=' +
                   name +
                   '&autocomplete=' +
                   str(params['autocomplete_setting']).lower() +
                   '&offset=' +
                   str(params['offset']) +
                   '&limit=' +
                   str(params['id_limit']) +
                   "&biolink_type=" +
                   params['biolink_type'])

    return nameres(itemRequest)


def normalize(item: str) -> Optional[IDList]:
    """
    Normalize a chemical identifier to get equivalent IDs.

    Args:
        item (str): Chemical identifier to normalize

    Returns:
        Optional[IDList]: List of equivalent IDs, or None if failed
    """
    print(f"Normalizing {item}...")
    item_request = (f"https://nodenormalization-sri.renci.org/1.5/get_normalized_nodes"
                   f"?curie={item}&conflate=true&drug_chemical_conflate=true"
                   f"&description=false&individual_types=false")

    failedCounts = 0

    while failedCounts < 5:
        try:
            response = requests.get(item_request, timeout=10)
            response.raise_for_status()
            output = json.loads(response.text)

            primary_key = list(output.keys())[0]
            label = output[primary_key]['id']['label']

            alternate_ids = output[primary_key]['equivalent_identifiers']
            returned_ids = [id_item['identifier'] for id_item in alternate_ids]

            return IDList(returned_ids)

        except Exception as e:
            print(f"Normalization attempt {failedCounts + 1} failed: {e}")
            failedCounts += 1

    return None


def n2s(instring: str) -> Optional[str]:
    """
    Convert chemical name to SMILES string.

    Args:
        instring (str): Chemical name

    Returns:
        Optional[str]: SMILES string if found, None otherwise
    """
    name_resolver_params = {
        "url": "https://name-resolution-sri.renci.org/",
        "service": "lookup",
        "autocomplete_setting": "true",
        "id_limit": "10",
        "offset": "0",
        "biolink_type": "ChemicalOrDrugOrTreatment",
    }

    try:
        id = identify(instring, name_resolver_params)
    except Exception as e:
        print(f"Identification failed: {e}")
        return None

    print(f"Found ID: {id}")

    alt_ids = normalize(id)

    if alt_ids is None:
        print("Normalization returned None")
        return None

    smiles = None
    if alt_ids.contains_pubchem():
        print("Found PubChem IDs in alt_ids")
        pubchem_ids = alt_ids.get_all_pubchem_ids()
        print(f"PubChem IDs: {pubchem_ids}")

        for item in pubchem_ids:
            try:
                print(f"Extracting SMILES from {item}")
                cid = int(item.replace("PUBCHEM.COMPOUND:", ""))
                smiles = get_smiles_from_pubchem(cid)

                if smiles:
                    print(f"Found SMILES: {smiles}")
                    return smiles
                else:
                    print(f"No SMILES found for {item}")

            except Exception as e:
                print(f"Error processing {item}: {e}")
                continue

    return None


# Example usage
if __name__ == "__main__":
    test_drugs = ["tenofovir", "aspirin", "ibuprofen"]

    for drug in test_drugs:
        print(f"\n{'='*60}")
        print(f"Testing: {drug}")
        print(f"{'='*60}")

        smiles = n2s(drug)

        if smiles:
            print(f"SUCCESS: {smiles}")
        else:
            print("FAILED: Could not find SMILES")
