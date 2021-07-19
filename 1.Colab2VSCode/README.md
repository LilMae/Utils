# Link Colab to VScode

**-This is method for connecting Colab to VScode-**


## 1. Run code below

-   You can result like this

<img src="media/5336b5fa68963d31980ac9682e7cc6c6.png" width="50%">


-   Go to link and get authorization code

-   Copy it and paste it to blank


## **2. Download Cloudflared**

-   Wait for downloading some utilities, and you can get result like this

<img src="media/9db22fb03b9fdb6ec4c301976ffe8787.png" width="50%">



-   Go to the link “Cloudflared (Argo Tunnel)”
<img src="media/06c7607bdafcfec4cb6508fd5fd791dd.png" width="50%">


-   Click “Install cloudflared”, and download file for your OS

-   Unzip the file and save it wherever you want


## **3. VScode setting**

-   Open VScode and download “Remote-SSH” extension
<img src="media/b469f75f3a1924490043befccd1e0106.png" width="50%">

-   And “Open a Remote Window”
<img src="media/44e15ed94dd8759868ce550b23a8ec7f.png" width="50%">

-   Click “Connect to Host…”
<img src="media/f63785a7fbcd58fdc150e3a104dd6477.png" width="50%">


-   Click “Configure SSH Hosts…” and click your config

-   The config will like this “C:\\Users\\Username\\.ssh\\config”
<img src="media/a624f9e5176c9256b756a39b12e05feb.png" width="50%">


-   You will get this config file
<img src="media/6b61ba2cd8f3965539ba3cf464902fc2.png" width="50%">



## **4. adding Host**

-   Go to Colab and copy this config info
<img src="media/5d78c47e59616758feb1cc3f88180096.png" width="50%">



-   Paste it to VScode config

-   Modify the ProxyCommand part to the location of the saved Cloudflared file
<img src="media/ea444862f6c2283664fab931d38e53d7.png" width="50%">


-   Save it!


## **5. Connect to Colab**

-   From Colab copy “VSCode Remote SSH”
<img src="media/5028bca0d10cdb1b3723fb83e4a158d8.png" width="50%">

-   Again, VScode-“Open remote Window”-“Connect to Host”

-   paste it
<img src="media/fbaa7e4dc5faec62e2f397cf1a1ac555.png" width="50%">


## **6. Remote Connection**

-   With opening new window, Some options are there
<img src="media/407093e055756aa7bd8489dd3bd4e2ee.png" width="50%">

-   First, Select Linux

-   and if there can be option asking “Continue”, just click “Continue”

-   Second, Enter password
<img src="media/835d931127b9b2c99b4f5b2a98a9c0a0.png" width="50%">

-   It will “test” as default
<img src="media/2187e26b2c39bd0734c8207cc3046969.png" width="50%">

-   Now, we just finished coneection!

-   Enjoy coding on VSCode with Colab
