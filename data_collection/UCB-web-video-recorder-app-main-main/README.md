# Audio-Video Speech Recorder
This web app features a Google Chrome browser speech recorder that connects to a client microphone, captures audio and video, and stores it on a server. The recording functionality has been extended to include video capture, enhancing the scope of the original audio-only feature.

The core client recording code for audio was forked from the open-source Vocaroo directory. The video recording functionality was developed as an extension of this base. This repository was originally forked from the recorder repo from getcontrast.io, before being adapted for research purposes. 

# Usage
## Configuration
You will need to deploy this on your own server and, if desired, host it on your own custom domain. The domain, if used, will require HTTPS from an SSL certificate, else the web app will not be able to request sufficient permissions to operate. You will need to input your server details in two separate places:
1) Within src > contexts > recording.tsx, there is an empty slot and comment specifiying where to insert your custom domain or IP address. 
2) Within src > components > Server.js, there is another empty slot where your desired endpoint should go. You will also need to configure your desired port, recordings and device information locations within this file (all marked by comments). 

The suggested web app deployment is then using Nginx, for which a config file can be used to specifiy external traffic redirection to the HTTPS domain. This config is user specific and is not provided as part of this repository. 

## Code
This code requires two app servers to run: a front-end client port (specified in your nginx configuration) and a backend server port(specified in the server.js code as above). Both must be running simultaneously for the app to function properly.

1. Navigate to the 'video-recorder-main' folder.
2. For the first run, install packages: yarn install.
3. Run the backend express server: node src/components/Server/server.js
4. Run the browser client server: yarn dev (dev mode).

This will open a browser window, and the app will be running. The recorded audio and video files will be saved to the desired locations specified in the server.js file as above, in the overall repository context. This 'uploads' folder can be blocked from upload by the .gitignore file, but please do take care to double-check that you are not including any recorded data in version controlling.

## Integration with Qualtrics or MTurk
A unique identifier is returned at the point of recording completion.
This identifier also names the file on the server.
Respondents can paste this back into the survey as a link to their file.

# Backlog
Create a single app start-up process to open both ports (instead of manually starting both).
Enable participant tracking across browser windows.
Host code on a VM and integrate with server management tools to protect against server failures.

# Contact
Please contact sbarrington@ischool.berkeley.edu for any queries.
