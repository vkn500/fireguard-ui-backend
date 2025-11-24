from ultralytics import YOLO

def run_val():
    # load your trained model
    model = YOLO("models/best_nano_111.pt")

    # run validation
    results = model.val(data="data/data.yaml", workers=0)

    # extract metrics
    metrics = results.results_dict

    map50 = metrics.get('metrics/mAP50(B)', None)          # mAP@0.5
    map5095 = metrics.get('metrics/mAP50-95(B)', None)     # mAP@0.5:0.95

    print("\n================ VALIDATION METRICS ================")
    print(f"mAP@0.5        : {map50 * 100:.2f}%")
    print(f"mAP@0.5:0.95   : {map5095 * 100:.2f}%")
    print("=====================================================")

if __name__ == "__main__":
    run_val()
