# Steps to Follow to Deploy App on EC2 (Ubuntu) Instance
In this scenario, the setup script is being attached during EC2 instance launch.
1. Navigate to the app directory
2. Make required changes to the code and the setup scripts:
    1. ```deploy-on-ec2.sh```
    2. ```ec2-setup.sh```
    3. ```create_psql_tables.sql```. We assume this an Ubuntu instance and that data files will be in ```/home/ubuntu/Phoso-AI/data/tables```. If not, 
    change path to ```S3``` bucket or other location.
3. Run the ```deploy-on-ec2.sh``` script
4. Verify if app is running locally
    1. Run this command:
    ```sudo systemctl status fastapi.service```

5. Update Twilio Webhook
    1. Grab instance IP adress 
    2. Make changes in Twilio Whatsapp sender 

6. Verify that Whatsapp app is working
    1. Send a test question: **Whats the price of Maize**
    2. Check app logs: ```sudo journalctl -u fastapi.service -f```