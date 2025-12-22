# Simple tool for exploratory data analysis

Simple app to conduct data analysis.

### How to use the tool

1. Visit https://simpletoolforexploratorydataanalysis.streamlit.app.
2. Click on Browse Files, under Ingest Data tab and select a local file to upload or insert a presigned url of a file on cloud storage, Onedrive, Google Drive.
3. If using Onedrive or Google Drive use the shared link as is.
4. Upon uploading or importing a csv, parquet or Excel file, the statitics/metadata of the file is automatically display on the Ingest Data tab.
5. To wrangle and pre-process the data if required, click the Data Transform tab. You can filter, group and remove columns amongst other things.
6. After processing the data if required, click on Create Charts tab.
7. Then click on Add Chart button, which brings up a form.
8. Fill the form, selecting any one of seven types of chart.
9. After completing the form, the Render Chart button appears.
10. Click the Render Chart button to be taken to the Dashboard page to see the chart.
11. Then click on Home on the sidebar.
12. Upload or import another file if required and cycle through the process again.
13. Should you want to use the same data, click on Transform Data tab and select Clear All Transforms under the Table Transform dropdown list.
14. If the the initial data processing is sufficient, just click on the Create Charts tab, click on Chart 1 expander and then click on Add Chart.
15. To view the transformed data at any time, click on Data Table tab.
16. A maximum of four charts can be created on the dashboard.
17. To ingest data from BiqQuery and Snowflake on your local machine, you must fork this code and create .streamlit/secrets.toml file in the root directory.
18. Paste the code below in the file with the appropriate parameters from Snowflake and or GCP.
19. The datanases and datasets parameters should be set to the databases and datasets on Snowflake and GCP you wish to be queried by the app.
20. If you want to deploy the app on Streamlit, please follow this https://docs.streamlit.io/develop/tutorials/databases/bigquery and https://docs.streamlit.io/develop/tutorials/databases/snowflake

<img width="940" height="499" alt="image" src="https://github.com/user-attachments/assets/0234613b-b512-4c43-8479-e996fcd09c12" />

<img width="940" height="499" alt="image" src="https://github.com/user-attachments/assets/fffbc45a-773e-4628-add5-57dbe979a6a0" />

<img width="940" height="499" alt="image" src="https://github.com/user-attachments/assets/1b6b7208-aebd-4567-8050-b1cb444105b9" />

<img width="940" height="499" alt="image" src="https://github.com/user-attachments/assets/48c3ffcd-cf39-4d61-84d0-ee5bb6147a02" />

<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/f92648b5-9b33-4483-adb4-e8daa28d21e2" />



```
[connections.snowflake]
account = "your_snowflake_account_identifier"
user = "your_snowflake_username"
password = "your_snowflake_password"
warehouse = "your_snowflake_warehouse"
databases = ["xxxx", "xxxx", ...]

[gcp_service_account]
type = "service_account"
project_id = "xxx"
private_key_id = "xxx"
private_key = "xxx"
client_email = "xxx"
client_id = "xxx"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "xxx"
datasets = ["xxxx", "xxxx", ...]
```


#### This is an on-going project
