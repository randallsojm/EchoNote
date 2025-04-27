const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const uploadBtn = document.getElementById('upload-btn'); // New Upload Button
const audioFileInput = document.getElementById('audioFileInput');
const loadingOverlay = document.getElementById('loading-overlay'); // Overlay element
const savedDocumentsHeader = document.getElementById('savedDocumentsHeader');
const documentsList = document.getElementById('documentsList');
let mediaRecorder;
let firstSoundTime; // Variable to store the time of the first sound (24-hour format)
let lastSoundTime;
let audioChunks = [];
let audioBlob = null;
let isUploading = false; // To prevent multiple uploads at the same time

// document.addEventListener("DOMContentLoaded", fetchSavedDocuments); // Fetch saved documents when the page loads
// when start button is pressed
// - access the users mic (ask for permission)
// - initialise MediaRecorder to record audio
// - start recording
// - whenever audio is detceted, it is added to the audioChunks list
// - when recording is stopped, create a Blob (audio file) and store it
// - hide start button and show stop button
startBtn.addEventListener('click', async () => {
    if (isUploading) return; // Prevent starting a new recording if upload is in progress
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
        if (audioChunks.length === 1) { // Check if this is the first audio chunk
            firstSoundTime = new Date().toLocaleTimeString('en-GB', {hour: '2-digit', minute: '2-digit'}); // Capture the current time in 24-hour format
            console.log("Time of first sound recorded (24-hour format):", firstSoundTime);
        }
    };

    // If you're not dealing with data specific to the event (like event.data), you don't need to use the event object.
    mediaRecorder.onstop = () => {
        // Once the recording stops, create a Blob (audio file) and store it
        audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        console.log("Audio Blob:", audioBlob); // Debugging
        audioChunks = [];
        //Audio Blob: Blob { size: 71979, type: "audio/wav" }
    
        uploadAudio(audioBlob);
    };

    // Start recording
    mediaRecorder.start();
    
    // Change button to stop recording
    startBtn.style.display = 'none'; // Hide start button
    stopBtn.style.display = 'inline-block'; // Show stop button
    stopBtn.disabled = false; // Ensure stop button is enabled
    uploadBtn.disabled = true;
});


stopBtn.addEventListener('click', () => {
    mediaRecorder.stop();
    // Hide the start, stop, and upload buttons and show the loading overlay
    lastSoundTime = new Date().toLocaleTimeString('en-GB', {hour: '2-digit', minute: '2-digit'}); // Capture the current time in 24-hour format
    startBtn.style.display = 'none'; // Hide start button
    uploadBtn.style.display = 'none'; // Hide upload button
    stopBtn.style.display = 'none'; // Hide stop button
    loadingOverlay.style.display = 'block'; // Show the overlay

});

// Upload Button's Event Listener
uploadBtn.addEventListener('click', () => {
    if (isUploading) return; // Prevent triggering upload if already uploading
    // Trigger the file input dialog
    firstSoundTime = 'N.A.';
    lastSoundTime = 'N.A.';
    audioFileInput.click();
});

// When the user selects a file, handle the file upload
audioFileInput.addEventListener('change', () => {
    const file = audioFileInput.files[0];
    if (file && !isUploading) {
        // Check if the MIME type is 'audio/x-wav', and if so, create a new Blob with 'audio/wav'
        if (file.type === 'audio/x-wav') {
            audioBlob = new Blob([file], { type: 'audio/wav' });
            console.log("Corrected MIME type to: 'audio/wav'");
            console.log("Audio Blob:", audioBlob);
        } else if (file.type === 'audio/mpeg') {
            audioBlob = new Blob([file], { type: 'audio/mpeg' });
            console.log("Corrected MIME type to: 'audio/mpeg'");
            console.log("Audio Blob:", audioBlob);
        } else {
            // For other file types, just use the file as it is (if necessary, add more types)
            audioBlob = file;
        }
        uploadAudio(audioBlob);
    } else {
        console.error("No audio file selected");
    }

    // Hide the start, upload, and stop buttons and show the loading overlay
    startBtn.style.display = 'none'; // Hide start button
    uploadBtn.style.display = 'none'; // Hide upload button
    stopBtn.style.display = 'none'; // Hide stop button
    loadingOverlay.style.display = 'block'; // Show the overlay
    audioFileInput.value = ''; // This clears the input field
}); 


// Upload function to send audio to backend
async function uploadAudio(audioBlob) {
    if (isUploading) return; // Prevent multiple uploads simultaneously
    isUploading = true; // Set uploading flag to true
    const formData = new FormData();
    const fileExtension = audioBlob.type === 'audio/mpeg' ? '.mp3' : '.wav'; // Determine file extension based on MIME type
    formData.append('audio', audioBlob, `recording${fileExtension}`);
    formData.append('firstSoundTime', firstSoundTime);
    formData.append('lastSoundTime', lastSoundTime);
    //'audio' is the key used to refer to the value in the form data, so on the server side, you can access this file using this key name (e.g., $_FILES['audio'] in PHP or req.files.audio in Node.js)

    try {
        const response = await fetch('http://localhost:3001/upload', {
            method: 'POST',
            body: formData
        });
    
        const result = await response.json();
        console.log("Server Response:", result);
    
        if (response.ok) {
            alert('Audio uploaded and processed successfully!');
            fetchSavedDocuments(); // Refresh saved documents after upload
        } else {
            alert(`Failed to process the audio. Server response: ${JSON.stringify(result)}`);
        }
    } catch (error) {
        console.error('Error uploading audio:', error);
        alert(`Error uploading audio: ${error.message}`);
    } finally {
        // Hide the loading overlay once upload is complete or on error
        loadingOverlay.style.display = 'none'; // Hide the overlay
        
        // Show the buttons again after upload is finished
        startBtn.style.display = 'inline-block'; // Show start button
        uploadBtn.style.display = 'inline-block'; // Show upload button
        stopBtn.style.display = 'none'; // Hide stop button
        isUploading = false; // Reset uploading flag
        uploadBtn.disabled = false;
    }
}

// Function to fetch and display saved documents


async function fetchSavedDocuments() {
    try {
        const response = await fetch('http://localhost:3001/documents'); // API to fetch saved docs
        // Sends an HTTP GET request to the backend at http://localhost:3001/documents to retrieve the list of processed documents.
        const data = await response.json();
        // Converts the received JSON response into a JavaScript object
        // data is now an object of an array of files
        console.log("Fetched documents:", data) 
       
        
        if (data.documents.length === 0) {
            savedDocumentsHeader.style.display = 'none'; // Hide if no documents
            documentsList.innerHTML = '<p>No documents found.</p>';
        } else {
            savedDocumentsHeader.style.display = 'block'; // Show if documents exist
            documentsList.innerHTML = ''; // Clear previous list

        //Loops through each document (doc) in the documents array
        data.documents.forEach(doc => {
            const docElement = document.createElement("div");
            docElement.classList.add("document-card");
            //document.createElement("div") → Creates a new <div> element for each document.
            //docElement.classList.add("document-card") → Adds the CSS class "document-card" to the element (likely used for styling)
            docElement.innerHTML = `
                <h3>${doc.title}</h3>
                <a href="http://localhost:3001/documents/${doc.filename}" target="_blank">View Document</a>
            `;
            //Creates a clickable link that points to http://localhost:3001/documents/${doc.filename}
            //target="_blank" opens the document in a new tab when clicked.
            documentsList.appendChild(docElement);
            //Adds the newly created docElement (containing title, summary, and link) to the documentsList container in the HTML.
            });
        }
    } catch (error) {
        console.error("Error fetching documents:", error);
    }
}

// The frontend calls GET /documents to get all processed files:

// { "documents": ["output1.txt", "summary2.txt"] }

// The user clicks on "output1.txt" in the frontend.
// The frontend should now request GET /documents/output1.txt to fetch or display it.