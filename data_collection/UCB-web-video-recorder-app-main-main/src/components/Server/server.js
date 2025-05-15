// server.js

const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const uuid = require('uuid');

const app = express();
const port = <PORT>; // INSERT YOUR DESIRED PORT HERE

const cors = require('cors');
app.use(cors());

// Utility function to ensure directory exists
function ensureDirectoryExists(dirPath) {
    if (!fs.existsSync(dirPath)) {
        console.log(`Creating directory: ${dirPath}`);
        fs.mkdirSync(dirPath, { recursive: true });
    }
}

console.log(__dirname);


// Serve static files from the client folder
app.use(express.static('../src'));

// Set up Multer storage configuration
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        // Extract userId from the file's original name
        const userId = file.originalname.split('_')[0];
        const userDir = path.join(__dirname, '<DESIRED RECORDINGS OUTPUT LOCATION>', userId); // INSERT YOUR DESIGGNED RECORDINGS OUTPUT FOLDER
        
        // Check if user specific directory exists; if not, create it
        if (!fs.existsSync(userDir)) {
            console.log(`Creating directory: ${userDir}`);  // Log the directory being created
            fs.mkdirSync(userDir, { recursive: true });  // The `recursive` flag ensures that 'uploads' is also created if it doesn't exist
        }
        
        cb(null, userDir); // Set the destination to the user specific directory
    },
    filename: (req, file, cb) => {
        ////cb(null, file.originalname.split('_').slice(1).join('_')); // Remove the userId from the filename
        const uniqueFilename = `${file.originalname}`;
        cb(null, uniqueFilename);
    }
});

const upload = multer({ storage: storage });

// File upload route
app.post('<ENDPOINT>', upload.single('recording'), (req, res) => { // INSERT YOUR ENDPOINT HERE INSTEAD OF <ENDPOINT>
    if (!req.file) {
        return res.status(400).send('No file was uploaded.');
    }
    
    // Save device type and microphone details to 'devices' folder
    const devicesDir = path.join(__dirname, '<DESIRED DEVICES OUTPUT LOCATION>'); // INSERT YOUR DESIRED DEVICES OUTPUT FOLDER
    ensureDirectoryExists(devicesDir);

    const userAgent = req.get('User-Agent');
    const microphoneDetails = req.body.microphone || "Microphone details not provided"; // Default message if microphone details are missing
    const userId = req.body.userId;
    fs.writeFileSync(path.join(devicesDir, `${userId}.txt`), `User-Agent: ${userAgent}\nMicrophone: ${microphoneDetails}`);

    res.send('Files uploaded successfully!');
});

// Start the server
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}/`);
});
