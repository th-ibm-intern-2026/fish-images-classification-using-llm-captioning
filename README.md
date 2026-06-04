# Fish Images Classification Using LLM Captioning

## Ingestion Pipeline

Use `main.py` in the `INGEST` folder.

- **Input**: CSV file with **3 columns**:  
  `[Fish Name, Summary Description, Image Links]`

- **What it Does**:
  - Embeds the `Summary Description`
  - Adds a new column: `embedding`
  - Stores all 4 columns in the Elasticsearch database

- **Output**:  
  None (but **remember the index name** — you'll need it for querying)

---

## Query Pipeline

### 🔹 How to Query
Use `main.py` in the `BE` folder.

- **Input**:
  1. Index name of the Elasticsearch DB used during ingestion
  2. Image path to search

- **What it Does**:
  - Transforms the image into 64x64
  - Generates a caption using WXAI
  - Embeds the caption
  - Queries the ES database using the embedded caption
  - Retrieves the top **N** matching fish

- **Output**:  
  JSON format of the top **N** fish results

---

## 📓 Example Service Usage

Check out [`service example.ipynb`](NOTEBOOKS/service_example.ipynb) in the `NOTEBOOKS` folder for more detail on ElasticsearchManager, ElasticsearchQuery and EmbeddingService

---

# Getting Environment Variables

After provisioning a new instance of Watson Discovery and other IBM Cloud services, you need to retrieve all the required environment variables.

## Required Environment Variables

### 1. Watson.ai Configuration

**PROJECT_ID**
- Project ID can be obtained after creating a project in watsonx.ai Runtime
- Don't forget to add watsonx.ai service to the project
![alt text](image.png)

**WATSONX_APIKEY**
- API key can be created with IAM API key service

**WATSONXAI_URL**
- Watson.ai URL can be found under the project (shown in step-1 image)
- Default: `https://us-south.ml.cloud.ibm.com`

**IAM_IBM_CLOUD_URL**
- Default: `iam.cloud.ibm.com`

**IBM_WATSONX_AI_INFERENCE_URL**
- Default: `https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29`

### 2. Elasticsearch Configuration

Elasticsearch credentials can be accessed within the Database for Elasticsearch product.

**es_endpoint**
![alt text](image-2.png)
- Endpoint can be accessed within the overview page of the product
- Replace username and password with actual credentials (see steps below)
- Format: `https://username:password@host:port`

**es_cert_path**
- Default: `/cert.pem`
- Certificate path for SSL connection

**es_username, es_password, and cert.pem**
![alt text](image-1.png)
- These credentials can be accessed via "Service credentials" in your Elasticsearch instance
- Create new credentials if none exist

### 3. Embedding Service Configuration

**EMBEDDING_SERVICE_URL**
- Get your embedding service URL from your deployment space (Code Engine)
- Make sure to include the correct endpoint path (e.g., `/extract_text`)
- Example: `https://your-embedding-service.appdomain.cloud/extract_text`

### 4. IBM Cloud Object Storage Configuration

**IBM_COS_API_KEY**
- Create an IAM API key for Cloud Object Storage access

**IBM_COS_RESOURCE_INSTANCE_ID**
- Go to Cloud Object Storage details to get the resource instance ID (CRN)
- This is the full CRN string starting with `crn:v1:bluemix:public:cloud-object-storage...`

**IBM_COS_ENDPOINT**
![alt text](image-3.png)
- After creating the `fish-image-bucket`, go to the overview to obtain the endpoint
- Example: `https://s3.us-south.cloud-object-storage.appdomain.cloud`

### 5. Code Engine Configuration

**CE_API_BASE_URL**
- Code Engine API base URLs are region-specific. Choose the region that matches your environment:

| Region | API Base URL |
|--------|-------------|
| US South (Dallas) | `https://api.private.us-south.codeengine.cloud.ibm.com` |
| US East (Washington DC) | `https://api.private.us-east.codeengine.cloud.ibm.com` |
| EU Germany (Frankfurt) | `https://api.private.eu-de.codeengine.cloud.ibm.com` |
| EU Great Britain (London) | `https://api.private.eu-gb.codeengine.cloud.ibm.com` |
| Japan (Tokyo) | `https://api.private.jp-tok.codeengine.cloud.ibm.com` |
| Japan (Osaka) | `https://api.private.jp-osa.codeengine.cloud.ibm.com` |
| Australia (Sydney) | `https://api.private.au-syd.codeengine.cloud.ibm.com` |

**Note:** Use the `.private` endpoints for secure, internal communication within IBM Cloud.

**CE_REGION**
- The region code for your Code Engine project (e.g., `us-south`, `jp-tok`, `eu-de`)

**CE_PROJECT_ID**
- Select your serverless project and open the details to see the GUID of the project
- This is the unique identifier for your Code Engine project

--

### Gemini API Key 
You can retrieve the Gemini Api Key from this [link](https://aistudio.google.com/).

## Example .env File

```env
# Watson.ai Configuration
PROJECT_ID=your-project-id-here
WATSONX_APIKEY=your-watsonx-api-key
WATSONXAI_URL=https://us-south.ml.cloud.ibm.com
IAM_IBM_CLOUD_URL=iam.cloud.ibm.com
IBM_WATSONX_AI_INFERENCE_URL=https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29

# Elasticsearch Configuration
es_endpoint=https://username:password@host:port
es_cert_path=/cert.pem
es_username=your-es-username
es_password=your-es-password

# Embedding Service
EMBEDDING_SERVICE_URL=https://your-embedding-service.appdomain.cloud/extract_text

# IBM Cloud Object Storage
IBM_COS_API_KEY=your-cos-api-key
IBM_COS_RESOURCE_INSTANCE_ID=crn:v1:bluemix:public:cloud-object-storage:global:...
IBM_COS_ENDPOINT=https://s3.us-south.cloud-object-storage.appdomain.cloud

# Code Engine Configuration
CE_API_BASE_URL=https://api.jp-tok.codeengine.cloud.ibm.com
CE_REGION=jp-tok
CE_PROJECT_ID=your-project-guid-here
GEMINI_API_KEY=<YOUR_GEMINI_API_KEY>
GROQ_API_KEY=<YOUR_GROQ_API_KEY>
```

## Testing Your Configuration

### Test Elasticsearch Connection
You can test your Elasticsearch connection using Postman:
1. Set request type to `GET`
2. URL: `https://your-elasticsearch-host:port`
3. Authorization: Basic Auth with your `es_username` and `es_password`
4. Should return cluster information if successful

### Test Embedding Service
1. Set request type to `POST`
2. URL: Your `EMBEDDING_SERVICE_URL`
3. Body (JSON): `{"sentence": ["test sentence"]}`
4. Should return embedding predictions if successful

## Notes
- Make sure all services are provisioned in the same IBM Cloud account
- Verify network connectivity between services if deploying to Code Engine
- Keep your credentials secure and never commit them to version control
- Ensure all region settings are consistent across your services


