var socket = io.connect(window.location.protocol + '//' + document.domain + ':' + location.port, {
    transports: ['websocket']
});

socket.on('connect', function () {
    console.log("Connected...!", socket.connected)
});

var canvas = document.getElementById('canvas');
var context = canvas.getContext('2d');
const video = document.querySelector("#videoElement");

video.width = 400;
video.height = 300;

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
setInterval(() => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    var data = canvas.toDataURL('image/jpeg', 0.5);
    context.clearRect(0, 0, canvas.width, canvas.height);
    socket.emit('image', data);
}, 1000 / FPS);

socket.on('processed_image', function (image) {
    photo.setAttribute('src', image);
});

socket.on('disconnect', function () {
    console.log("Disconnected from the server");
});
