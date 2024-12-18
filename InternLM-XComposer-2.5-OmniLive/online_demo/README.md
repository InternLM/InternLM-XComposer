# **InternLM-XComposer-2.5-OmniLive Setup Guide**

To set up XComposer-2.5-OL, deploy the following three components:

1. **Frontend**: Captures audio and video data and sends it to the SRS server.

2. **SRS Server**: Converts raw audio and video data into a streamable format.

3. **Backend**: Requests streaming data from the SRS server, processes it, and sends the response back to the frontend.

## **Deployment Guidelines**

Follow the steps below to deploy these components. This setup has been tested only when all components are **_within the same local network_**. Using components across different networks may cause connectivity issues.

### **SRS Server**

1.	Execute the following commands in your terminal to run the server. Replace 192.168.3.10 with your machine’s local network IP address (do not use 127.0.0.1).

```shell
export CANDIDATE="192.168.3.10" # Ensure this is the LAN address, not 127.0.0.1
docker run --rm --env CANDIDATE=$CANDIDATE \
  -p 1935:1935 -p 8080:8080 -p 1985:1985 -p 8000:8000/udp \
  registry.cn-hangzhou.aliyuncs.com/ossrs/srs:5 \
  objs/srs -c conf/rtc2rtmp.conf
```

2. Verify the SRS server’s functionality:
- Open http://localhost:8080/players/rtc_publisher.html?autostart=true&stream=livestream&schema=http.
- In the RTC Streaming tab, click Stream.
- Open http://localhost:8080/players/whep.html and check if the stream plays.

### **Backend**

The backend can be deployed on a local machine or a remote server. 

1. Download the Model

```shell
cd InternLM-XComposer/InternLM-XComposer-2.5-OmniLive
huggingface-cli download internlm/internlm-xcomposer2d5-ol-7b \
  --local-dir internlm-xcomposer2d5-ol-7b \
  --local-dir-use-symlinks False \
  --resume-download
```

2. Get the merged lora model
 
 ```shell
python examples/merge_lora.py
```

3. Change the model path in the [start script](Backend/backend_ixc/start.sh)

```shell
export ROOT_DIR=$Your_Download_Model_Path
```

4. Start the backend with the modified start script:

```shell
cd online_demo/Backend/backend_ixc
sh start.sh
```

### **Frontend**

1.	Refer to the [frontend setup instructions](Frontend/README.md).
2.	If the backend is deployed on a remote server, replace localhost in [CHAT_SOCKET_URL](Frontend/src/config/service-url.ts) with the server’s IP address.
3.	Start the frontend using the following command:

```shell
npm run start
```

Once all components are properly deployed, you can begin using XComposer-2.5-OL.