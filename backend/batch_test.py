import os
import time
import requests

# ======================
# CONFIG
# ======================
BASE_URL = "http://127.0.0.1:8000"
PDF_FOLDER = "test cases"
OUTPUT_FOLDER = "outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Summary modes
summary_modes = ["snapshot", "overview", "deepdive"]

print("STARTED - GENERATION ONLY (test1 → test6)")

# ======================
# GENERATE ONLY
# ======================
for i in range(1, 7):
    file_name = f"test{i}.pdf"

    print(f"\n=== Processing {file_name} ===")

    # ======================
    # GENERATE SUMMARIES
    # ======================
    for mode in summary_modes:
        print(f"  → {mode}")

        try:
            res = requests.post(
                f"{BASE_URL}/send",
                json={
                    "message": "summarize",
                    "document": file_name,
                    "mode": "summary",
                    "summary_mode": mode
                },
                timeout=300
            )

            if res.status_code != 200:
                print(f"❌ Failed {mode}")
                continue

            reply = res.json().get("reply", "")

            # Create folder per PDF
            pdf_output_folder = os.path.join(OUTPUT_FOLDER, f"test{i}")
            os.makedirs(pdf_output_folder, exist_ok=True)

            # Save file
            output_path = os.path.join(pdf_output_folder, f"{mode}.txt")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(reply)

            print(f"✔ Saved {mode}")

        except Exception as e:
            print(f"❌ Error in {mode}: {e}")

        time.sleep(1)

    print(f"✔ Done {file_name}")

    time.sleep(2)

print("\n🎉 ALL DONE")