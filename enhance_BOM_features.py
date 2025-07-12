# analyze_bom_structure.py

import pandas as pd

def analyze_bom(df):
    # print available columns
    print("Available columns in the DataFrame:")
    print(df.columns.tolist())
    # Step 1: Mark IsAssembly (True if Component appears as a Next Assembly anywhere else)
    next_assemblies = set(df['Next Assembly'].unique())
    df['IsAssembly'] = df['Component'].apply(lambda x: x in next_assemblies)

    # Step 2: Create a mapping from Component to WBS Code
    component_to_wbs = dict(zip(df['Component'], df['WBS Code']))

    # Step 3: Create Parent WBS Code column (get the WBS Code of the Next Assembly this row rolls up into)
    df['Parent WBS Code'] = df['Next Assembly'].map(component_to_wbs).fillna('')

    # Step 4: Find possible children for Level 2 WBS Codes
    # find unique Level 2 WBS Codes (WBS_level == 2)
    level_2_wbs_codes = df[df['WBS_level'] == 2]['WBS Code'].unique().tolist()
    children_by_wbs = {}
    for parent in level_2_wbs_codes:
        children = df[df['Parent WBS Code'] == parent]['WBS Code'].unique().tolist()
        children_by_wbs[parent] = sorted(set(children))

    # make dictionary of WBS descriptions by WBS Code
    # trim spaces from WBS Description
    df['WBS Description'] = df['WBS Description'].str.strip()
    wbs_descriptions = dict(zip(df['WBS Code'], df['WBS Description']))

    # Print child WBS relationships in format WBS Code: WBS Description
    for parent, children in children_by_wbs.items():
        print(f"\nPossible children of WBS Code {parent} ({wbs_descriptions.get(parent, 'Unknown')}):")
        for child in children:
            print(f"  - {child} ({wbs_descriptions.get(child, 'Unknown')})")

    return df, children_by_wbs

if __name__ == '__main__':
    #load the BOM data into DF if not already loaded
    df = pd.read_csv('bom_data.csv')
    enriched_df, child_wbs_dict = analyze_bom(df)
    enriched_df.to_csv('bom_with_structure.csv', index=False)
