import pandas as pd
import os

# --- Configuration - Adjust these paths as necessary ---

LAN_DATA_LONG_FORMAT_CSV = "data/processed/lan_data_long_format.csv"
SCRATCH_CSV_INPUT = "data/processed/scratch.csv" # Make sure this is correct if scratch is in wilcox/data/processed/
SCRATCH_CSV_OUTPUT = "data/processed/scratch_augmented.csv"

# Column names
# Adjust these if your column names are different in lan_data_long_format.csv
ITEM_COL_NAME_IN_LAN_DATA = "item_id" # Or "item_id" etc.
CONDITION_COL_NAME_IN_LAN_DATA = "condition"

# Name for the critical word column from your scratch.csv
# If your scratch.csv has no header, pandas reads it as column '0'.
# If it has a header (e.g., 'critical_region'), use that.
CRITICAL_WORD_COL_NAME_IN_SCRATCH = "critical_region" # Or 'critical_region' or 0 if no header

# --- Main script logic ---
def augment_scratch_file():
    print(f"Loading LAN data from: {LAN_DATA_LONG_FORMAT_CSV}")
    if not os.path.exists(LAN_DATA_LONG_FORMAT_CSV):
        print(f"Error: File not found: {LAN_DATA_LONG_FORMAT_CSV}")
        return
    try:
        lan_df = pd.read_csv(LAN_DATA_LONG_FORMAT_CSV)
    except Exception as e:
        print(f"Error loading {LAN_DATA_LONG_FORMAT_CSV}: {e}")
        return

    print(f"Loading scratch data from: {SCRATCH_CSV_INPUT}")
    if not os.path.exists(SCRATCH_CSV_INPUT):
        print(f"Error: File not found: {SCRATCH_CSV_INPUT}")
        return
    try:
        # If your scratch.csv has NO header for the critical words column:
        # scratch_df = pd.read_csv(SCRATCH_CSV_INPUT, header=None, names=[CRITICAL_WORD_COL_NAME_IN_SCRATCH])
        # If your scratch.csv HAS a header:
        scratch_df = pd.read_csv(SCRATCH_CSV_INPUT)
        # If using a specific column name and it's not the one read by default:
        # You might need to rename it if it's just one col, e.g. scratch_df.rename(columns={scratch_df.columns[0]: CRITICAL_WORD_COL_NAME_IN_SCRATCH}, inplace=True)

    except Exception as e:
        print(f"Error loading {SCRATCH_CSV_INPUT}: {e}")
        return

    # Verify expected columns exist
    if ITEM_COL_NAME_IN_LAN_DATA not in lan_df.columns:
        print(f"Error: Column '{ITEM_COL_NAME_IN_LAN_DATA}' not found in {LAN_DATA_LONG_FORMAT_CSV}")
        return
    if CONDITION_COL_NAME_IN_LAN_DATA not in lan_df.columns:
        print(f"Error: Column '{CONDITION_COL_NAME_IN_LAN_DATA}' not found in {LAN_DATA_LONG_FORMAT_CSV}")
        return
    if CRITICAL_WORD_COL_NAME_IN_SCRATCH not in scratch_df.columns and not (isinstance(CRITICAL_WORD_COL_NAME_IN_SCRATCH, int) and CRITICAL_WORD_COL_NAME_IN_SCRATCH < len(scratch_df.columns)):
        print(f"Error: Column '{CRITICAL_WORD_COL_NAME_IN_SCRATCH}' not found in {SCRATCH_CSV_INPUT}. Actual columns: {scratch_df.columns.tolist()}")
        # If CRITICAL_WORD_COL_NAME_IN_SCRATCH was intended to be, e.g. 'critical_region' from user example
        # and user confirmed it's the only column, we can assume it's scratch_df.columns[0]
        # For simplicity, ensure CRITICAL_WORD_COL_NAME_IN_SCRATCH matches the actual header or is 0 for headerless.
        return


    if len(lan_df) != len(scratch_df):
        print(f"Warning: Row count mismatch! lan_data_long_format.csv has {len(lan_df)} rows, scratch.csv has {len(scratch_df)} rows.")
        print("Proceeding, but concatenation might be misaligned if counts differ.")
        # Decide if you want to stop: return

    # Create the new DataFrame
    # Select the required columns from lan_df
    items_and_conditions = lan_df[[ITEM_COL_NAME_IN_LAN_DATA, CONDITION_COL_NAME_IN_LAN_DATA]].copy()
    
    # Get the critical word column from scratch_df
    critical_words_column = scratch_df[[CRITICAL_WORD_COL_NAME_IN_SCRATCH]].copy()

    # Reset index for robust concatenation if they weren't already default 0-N
    items_and_conditions.reset_index(drop=True, inplace=True)
    critical_words_column.reset_index(drop=True, inplace=True)

    # Concatenate them side-by-side
    augmented_df = pd.concat([items_and_conditions, critical_words_column], axis=1)

    # Ensure column names are as expected for the next script
    # This step renames the columns to 'item', 'condition', 'actual_critical_word'
    # which the aggregate_surprisals.py modification expects.
    augmented_df.rename(columns={
        ITEM_COL_NAME_IN_LAN_DATA: 'item',
        CONDITION_COL_NAME_IN_LAN_DATA: 'condition',
        CRITICAL_WORD_COL_NAME_IN_SCRATCH: 'actual_critical_word'
    }, inplace=True)


    try:
        augmented_df.to_csv(SCRATCH_CSV_OUTPUT, index=False)
        print(f"Successfully created augmented scratch file: {SCRATCH_CSV_OUTPUT}")
        print(f"It has columns: {augmented_df.columns.tolist()}")
        print(f"First few rows:\n{augmented_df.head()}")
    except Exception as e:
        print(f"Error saving augmented scratch file: {e}")

if __name__ == "__main__":
    # Before running, make sure:
    # 1. Your `scratch.csv` (SCRATCH_CSV_INPUT) contains ONE column with the critical words.
    #    If it has a header, set CRITICAL_WORD_COL_NAME_IN_SCRATCH to that header name.
    #    If it has NO header, pandas will read the column as `0`. In that case, either set
    #    CRITICAL_WORD_COL_NAME_IN_SCRATCH = 0 (the integer)
    #    OR use the `header=None, names=[...]` option in `pd.read_csv` for scratch_df.
    #    For example, if scratch.csv has no header:
    #    scratch_df = pd.read_csv(SCRATCH_CSV_INPUT, header=None, names=['my_critical_words'])
    #    CRITICAL_WORD_COL_NAME_IN_SCRATCH = 'my_critical_words' # then use this name
    
    # For your example "critical_region \n soon \n soon...",
    # it implies 'critical_region' is the header. So set:
    # CRITICAL_WORD_COL_NAME_IN_SCRATCH = "critical_region" 
    # And ensure your scratch.csv file actually has "critical_region" as the first line.

    augment_scratch_file()