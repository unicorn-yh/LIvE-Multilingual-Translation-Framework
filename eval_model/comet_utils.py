from comet.models import download_model, load_from_checkpoint


# Load the COMET model
def init_comet():
    model_path = "/Users/olivia/Project/Gemma-Multilingual-Model/eval/comet/checkpoints/model.ckpt"  # Downloads the pre-trained COMET model
    model = load_from_checkpoint(model_path)
    return model

# Define your source, hypothesis, and reference sentences
source = "This is the source sentence."
hypothesis = "This is the hypothesis sentence."
reference = "This is the reference sentence."

# Prepare data in the format required by COMET
data = [
    {
        "src": source,
        "mt": hypothesis,
        "ref": reference,
    }
]

def comet_score(model, data):
    # Score the data
    scores = model.predict(data, batch_size=1, gpus=1)
    for item in scores:
        if item[0] == 'system_score':
            return item[1]  # Return the system_score directly

    # Optional: return None or raise an error if 'system_score' is not found
    return 0

    # Display the COMET score
    # print(f"COMET Score: {scores[0]}")
