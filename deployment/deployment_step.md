## Deploying a Custom TensorFlow Model on Google Cloud Vertex AI

This guide provides step-by-step instructions to deploy a custom-trained TensorFlow model (trained outside GCP) on Google Cloud Vertex AI for serving at scale using Vertex AI Endpoints.

### Prerequisites

- A TensorFlow model trained outside of GCP (downloaded from Kaggle output)
- A Google Cloud Platform (GCP) account
- Google Cloud SDK installed locally or using Google Cloud Shell for testing endpoint
- Access to Google Cloud Console
- GCP project with billing enabled

### Step 1: Download Your Trained Model

After training your TensorFlow model on Kaggle (or any other platform), download the model files. Ensure your model is saved in TensorFlow's SavedModel format (`.pb`, `variables/`, etc.).

Example Kaggle output path:  
`/kaggle/working/{your_model_folder}`

You may use my model output for testing: '{movie-review-classifier/}'
www.kaggle.com/code/arjunraghunandanan/fellowship-ai-nlp-challenge-imdb50k-tf-text/output

### Step 2: Create a Google Cloud Storage Bucket

1. Log in to your GCP account.
2. Navigate to [Google Cloud Console](https://console.cloud.google.com/).
3. In the side menu, go to **Storage** > **Browser**.
4. Click **Create Bucket**.
5. Provide a unique name for your bucket, choose your region, and configure any other necessary settings.
6. Click **Create**.

### Step 3: Upload Your Model to the Cloud Bucket

1. Once your bucket is created, navigate to it in the Cloud Console.
2. Click **Upload files** and select your TensorFlow model files.
3. After the upload completes, note the path to your model in Google Cloud Storage (e.g., `gs://{bucket-name}/{model-folder}`).

### Step 4: Import the Model into Vertex AI

1. Navigate to the [Vertex AI Models Console](https://console.cloud.google.com/vertex-ai/models?).
2. Click the **Import** button.
3. Choose **Import new model**.
4. Enter the following details:
   - **Model Name**: A name for your model (e.g., `custom-tensorflow-model`).
   - **Description**: A brief description of the model.
   - **Region**: Select the region where you want the model to be deployed (ensure it's the same as your bucket’s region).
5. In the Import Model from Google Cloud Storage section:
   - Select the model files from your Google Cloud Storage bucket (the `gs://` path noted earlier).
   - Configure the model with any additional settings.
6. Click **Import**.

   Wait for the model to fully upload and be imported into Vertex AI. This may take some time depending on the size of your model.

### Step 5: Create an Endpoint for Online Predictions

1. Once the model is imported, navigate to the [Vertex AI Endpoints Console](https://console.cloud.google.com/vertex-ai/online-prediction/endpoints?hl=en&project={your_project_id}).
2. Click **Create Endpoint**.
3. Provide a name for the endpoint (e.g., `custom-model-endpoint`).
4. Select the model you just imported.
5. Choose the appropriate machine type for your endpoint. Consider the expected traffic and computational needs. You can select from different machine types based on your workload, such as:
   - `n1-standard-4` (general-purpose, balanced)
   - `n1-highmem-8` (high-memory workloads)
   - `n1-highcpu-16` (compute-heavy workloads)
6. Configure autoscaling settings, if needed, based on the expected demand.
7. Click **Deploy** and wait for the endpoint to be created.

### Step 6: Make Predictions Using the Deployed Model

Once the endpoint is deployed, you can start making online predictions. Use the endpoint URL provided in the Vertex AI console or make API requests using GCP’s Vertex AI SDK.

Check inside webapp/ folder in this repo to see my app. Note that calling endpoint required authorization tokens or other methods. Refer to the model sample request section in the endpoint testing page to see which option suits your need. 

Ensure that you have appropriate IAM permissions to access Vertex AI and deploy models. Also ensure you have all recommended APIs enabled
You can monitor the model's performance, latency, and other metrics in the Monitoring section of Vertex AI.
For more details on pricing, check the Vertex AI pricing page.
