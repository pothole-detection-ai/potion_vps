// var socket = io.connect('wss://vps.potion.my.id', {
//     transports: ['websocket']
// });

var socket = io.connect('http://127.0.0.1:5000/', {
    transports: ['websocket']
});

socket.on('connect', function () {
    console.log("Connected...!", socket.connected);
});

socket.on('error', function (error) {
    console.error("Socket error:", error);
});

socket.on('processed_image', function (image) {
    console.log("Received new frame");
    photo.setAttribute('src', image);
});

var canvas = document.getElementById('canvas');
var context = canvas.getContext('2d');
const video = document.querySelector("#videoElement");

// Adjust target width and height based on your requirements
const targetWidth = 400;
const targetHeight = 300;

video.width = targetWidth;
video.height = targetHeight;

if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({
        video: true
    })
        .then(function (stream) {
            video.srcObject = stream;
            video.play();
        })
        .catch(function (error) {
            console.error("Error accessing camera:", error);
        });
}

const FPS = 10;
const frameBuffer = [];

function updatePhoto(image) {
    frameBuffer.push(image);

    if (frameBuffer.length > BUFFER_SIZE) {
        frameBuffer.shift();
    }

    renderFrames();
}

function renderFrames() {
    if (frameBuffer.length > 0) {
        const nextFrame = frameBuffer.shift();
        photo.setAttribute('src', nextFrame);
        requestAnimationFrame(renderFrames);
    }
}

setInterval(() => {
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    context.drawImage(video, 0, 0, targetWidth, targetHeight);

    var data = canvas.toDataURL('image/jpeg', 0.8); // Adjust compression quality if needed
    context.clearRect(0, 0, canvas.width, canvas.height);

    socket.emit('image', data);
}, 2000);

