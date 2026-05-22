import sys
from server import predict_plant_disease

def test_local_prediction():
    print("Testing plant disease prediction tool using the locally generated diseased tomato leaf image...")
    image_path = "/home/aishwarya/.gemini/antigravity-ide/brain/715c9b17-79c6-4040-83c4-28c87fbc0984/tomato_leaf_disease_1779434860035.png"
    
    try:
        report = predict_plant_disease(image_data=image_path)
        print("\n--- ASSESSMENT REPORT ---")
        print(report)
        print("-------------------------\n")
        
        # Verify the report contents
        if "Plant Disease Assessment Report" in report and "Error" not in report:
            print("SUCCESS: The tool was successfully invoked and returned a valid report!")
            return True
        else:
            print(f"FAILURE: The report contains an unexpected result or error: {report}")
            return False
    except Exception as e:
        print(f"FAILURE: Test execution encountered an exception: {e}")
        return False

if __name__ == "__main__":
    success = test_local_prediction()
    if not success:
        sys.exit(1)

