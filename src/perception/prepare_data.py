import os
import sys

# Let's add the project root to the path to avoid import errors
# if you run the script from the src/perception/ folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.perception.data_utils import calculate_and_save_stats

def main():
    # Configuration of paths
    # We assume the script is run from the root of the Agri-Swarm project\
    data_directory = "data/raw"
    output_json = "data/processed/dataset_stats.json"
    
    print("🚀 Starting data preparation for Agri-Swarm...")
    
    # Verify that the data folder exists
    if not os.path.exists(data_directory):
        print(f"❌ Error: The folder '{data_directory}' does not exist.")
        print("Make sure you have extracted the dataset in data/raw/train and data/raw/test.")
        return

    try:
        # Call to the function in data_utils
        stats = calculate_and_save_stats(
            data_dir=data_directory, 
            output_path=output_json
        )
        
        print("\n--- Results ---")
        print(f"📍 File saved: {output_json}")
        print(f"📊 Mean (RGB): {stats['mean']}")
        print(f"📉 Standard Deviation (RGB): {stats['std']}")
        print(f"🖼️ Image Size: {stats['img_size']}")
        print("\n✅ Dataset statistics calculation completed successfully!")
        
    except Exception as e:
        print(f"💥 An error occurred during the calculation: {e}")

if __name__ == "__main__":
    main()