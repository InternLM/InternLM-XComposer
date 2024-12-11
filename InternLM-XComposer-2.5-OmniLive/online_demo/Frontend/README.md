React front-end project integrated with [WebSocket](https://developer.mozilla.org/zh-CN/docs/Web/API/WebSocket), [WebRTC](https://developer.mozilla.org/zh-CN/docs/Web/API/WebRTC_API) and [SRS](https://github.com/ossrs/srs).

# Notice

-   Navigating away from the page (making it invisible) and then returnning will trigger a websocket reconnection.
-   The project requires Node.js `version >= 18.0.0`.

# Prepare your front-end development environment

[Node.js](https://nodejs.org/en)® is a free, open-source, cross-platform JavaScript runtime environment that lets developers create servers, web apps, command line tools and scripts.

## Node.js Installation Guide (Windows, Linux, macOS)

### Windows Installation

-   **Step 1: Download Node.js**

    -   Open your web browser and visit the [Node.js official website](https://nodejs.org/en).

    -   Navigate to the "Downloads" section.

    -   Select the desired version (LTS recommended for long-term stability). As of November 2024, the latest LTS version might be v22.x.x.

    -   Click on the "Windows Installer (.msi)" link to download the installation package.

-   **Step 2: Install Node.js**

    -   Double-click the downloaded .msi file to start the installation wizard.

    -   Click "Next" to proceed.

    -   Read and accept the license agreement by checking the "I accept the terms in the License Agreement" box.

    -   Click "Next" again and select the installation directory. It's recommended to change the default location to avoid installing in the C drive.

    -   Continue clicking "Next" to use the default settings until you reach the "Install" button.

    -   Click "Install" to start the installation process.

    -   Wait for the installation to complete and click "Finish" to exit the installation wizard.

-   **Step 3: Verify Installation**

    -   Open the Command Prompt (cmd) by pressing `Win + R`, typing `cmd`, and pressing Enter.
    -   Type `node -v` and press Enter. You should see the installed Node.js version displayed.
    -   Type `npm -v` and press Enter to verify the installed npm version. Npm is the package manager that comes bundled with Node.js.

-   **Step 4: Configure npm Global Path (Optional)**
    If you want to change the default global installation path for npm, follow these steps:

    -   Open the Command Prompt (cmd) as an administrator.

    -   Navigate to your Node.js installation directory (e.g., C:\Program Files\nodejs).

    -   Create two new folders named node_global and node_cache.

    -   Run the following commands to set the new paths:

        ```bash
        npm config set prefix "C:\Program Files\nodejs\node_global"
        npm config set cache "C:\Program Files\nodejs\node_cache"
        ```

    -   Open the Environment Variables settings in the System Properties.
    -   Add `C:\Program Files\nodejs\node_global` to the `PATH` variable under User Variables.
    -   Optionally, create a new system variable named `NODE_PATH` and set its value to ` C:\Program Files\nodejs\node_global\node_modules`.

### Linux Installation

-   **Step 1: Update Your System**
    Before installing Node.js, ensure your Linux system is up-to-date:

    ```bash
    sudo apt-get update
    sudo apt-get upgrade
    ```

-   **Step 2: Install Dependencies**
    Node.js requires certain dependencies to function properly:

    ```bash
    sudo apt-get install build-essential libssl-dev
    ```

-   **Step 3: Download and Install Node.js**
    You can download the Node.js source code or use a package manager like `curl` or `wget` to download a pre-built binary. For simplicity, this guide assumes you're using a package manager.

    -   Navigate to the Node.js download page for package managers.
    -   Follow the instructions for your Linux distribution. For example, on Ubuntu, you can use:

        ```bash
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        sudo apt-get install -y nodejs
        ```

    -   Replace 20.x with the desired version number if you don't want the latest version.

-   **Step 4: Verify Installation**
    -   Open a terminal.
    -   Type `node -v` and press Enter to check the Node.js version.
    -   Type `npm -v` and press Enter to verify the npm version.

### MacOS Installation

Installing Node.js on macOS is a straightforward process that can be accomplished using the official installer from the Node.js website or through package managers like Homebrew. This guide will cover both methods.

#### Method 1: Using the Official Installer

-   Visit the Node.js Website
    -   Open your web browser and navigate to https://nodejs.org/.
-   Download the Installer
    -   Scroll down to the "Downloads" section.
    -   Click on the "macOS Installer" button to download the .pkg file. Ensure you download the latest version, which as of August 2024, might be v20.x.x or higher.
-   Install Node.js
    -   Once the download is complete, locate the .pkg file in your Downloads folder.
    -   Double-click the file to start the installation process.
    -   Follow the on-screen instructions. Typically, you'll need to agree to the license agreement, select an installation location (the default is usually fine), and click "Continue" or "Install" until the installation is complete.
-   Verify the Installation
    -   Open the Terminal application by going to "Finder" > "Applications" > "Utilities" > "Terminal" or using Spotlight Search (press `Cmd + Space` and type "Terminal").
    -   Type `node -v` and press Enter. This command should display the installed version of Node.js.
    -   Type `npm -v` and press Enter to verify that npm, the Node.js package manager, is also installed.

#### Method 2: Using Homebrew

If you prefer to use a package manager, Homebrew is a popular choice for macOS.

-   Install Homebrew (if not already installed)

    -   Open the Terminal.

    -   Copy and paste the following command into the Terminal and press Enter:

        ```bash
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        ```

    -   Follow the on-screen instructions to complete the Homebrew installation.

-   Install Node.js with Homebrew
    -   Once Homebrew is installed, update your package list by running brew update in the Terminal.
    -   To install Node.js, run the following command in the Terminal:
        ```bash
        brew install node
        ```
    -   Homebrew will download and install the latest version of Node.js and npm.
-   Verify the Installation
    -   As with the official installer method, you can verify the installation by typing node -v and npm -v in the Terminal and pressing Enter.

#### Additional Configuration (Optional)

-   Configure npm's Global Installation Path (if desired):
    -   You may want to change the default location where globally installed npm packages are stored. Follow the steps outlined in the Node.js documentation or search for guides online to configure this.
-   Switch to a Different Node.js Version (if needed):
    -   If you need to switch between multiple Node.js versions, consider using a version manager like nvm (Node Version Manager). Follow the instructions on the nvm GitHub page to install and use it.

By following these steps, you should be able to successfully install Node.js on your system. Remember to keep your Node.js and npm versions up-to-date to take advantage of the latest features and security updates.

If your env has been prepared, you can

# Installation and Setup Instructions

## Installation

```
  npm install
```

## Start Server

```
  npm start
```

## Visit Server

```
  https://localhost:8081
```

-   Pay attention to the real port in your terminal.maybe it won`t be 8081.
-   When you open the webpage, it may prompt "Not Secure", and you need to choose "Continue to this website".
-   When the webpage opens, pay attention to the browser's prompt regarding openning the camera and enable the local camera.


# Configuration

## Conversation Implementation Process

The process for implementing real-time conversation on the front-end involves serveral key steps:

- Open local camera: call JS API `navigator.mediaDevices.getUserMedia` to get local audio and video stream.
- Initiate stream pushing: call http request to SRS server to initiate stream pushing.
- Establish WebRTC connection: Set up a WebRTC connection to facilitate the transmission of audio and video streams.
- Open WebSocket: Once the stream is successfully pushed, open a WebSocket to enable bidirectional communication for voice input and model responses.


## How to modify the request URL

There are two requests for the whole project, which are defined in `src/config/service-url.ts`

-   `SRS_BASE_URL`: SRS service url, based on `WebRTC` protocol, is used for video stream pushing and publishing.
-   `CHAT_SOCKET_URL`: Chat service url, base on `WebSocket` protocol, is used for session message transmission.

You can modify these urls, and the requests on the local page will refresh immediately.

**Pay attention to the following points:**

The request initiated for "Initiate stream pushing" is an HTTP request with the path `/rtc/v1/publish`. Due to security considerations, browsers impose cross-origin restrictions, preventing local front-end services from directly accessing service addresses with different protocols, ports, or domain names. Therefore, a proxy needs to be configured, with the target being the address of the SRS service. The proxy configuration can be modified in `vite.config.ts`.

```
server: {
    port: 8081,
    proxy: {
        '/rtc': {
            target: 'http://localhost:1985', // Modify this line to change the target port of SRS service
            changeOrigin: true,
        },
    },
},
```