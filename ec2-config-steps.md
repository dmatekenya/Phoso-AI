# Steps to Follow to Deploy App on EC2 (Ubuntu) Instance
## Manual Configuration
In this scenario, the setup script is not uploaded while launching the instance
and so these steps are run manually
1. Navigate to the app directory
3. Make any changes to the code and the setup scripts:
    - ```deploy-on-ec2.sh```
    - ```ec2-setup.sh```
2. Run the ```deploy-on-ec2.sh``` script
3. Connect to EC2 instance and follpw the steps below.

    - Install AWS CLI
        - ```sudo apt install unzip```
        - ```curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o     "awscliv2.zip" unzip awscliv2.zip sudo ./aws/install```

    - Download ```ec2-setup.sh``` script from s3 onto instance:```aws s3 cp s3://chichewa-ai/phoso-ai-files/ec2-setup.sh .```

    - Make the script executable: ```chmod +x ec2-setup.sh```

    - Run the script
5. Verify if app is running
    1. Check by sending WhatsApp message to Chatbot to see if its working

## Automatic Configuration


