const express = require("express");
const path = require("path");
const app = express();
const http = require("http");
const server = http.createServer(app);
const cors = require("cors");
const bodyParser = require("body-parser");
const { Server } = require("socket.io");
const io = new Server(server);
// var chat = require("./chat");
// chat(io);

require("@tensorflow/tfjs-backend-cpu");
require("@tensorflow/tfjs-backend-webgl");
require("@tensorflow/tfjs-node");
const ObjectDetectors = require("./object_detector/ObjectDetectors");
const cocoSsd = require("@tensorflow-models/coco-ssd");
const PORT = process.env.PORT || 5000;

app.use(bodyParser.json({ limit: "50mb" }));
app.use(express.static(path.join(__dirname, "build")));
app.use(
  bodyParser.urlencoded({
    limit: "50mb",
    extended: true,
    parameterLimit: 50000,
  })
);
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: false }));
app.use(cors());

app.get("/", (req, res) => {
  console.log("== User connected to /home");
  res.json("Hello there, express server started");
});

app.post("/image-upload", async (req, res) => {
  console.log("== User uploaded image to coco-ssd");
  const data = req.body;
  const ObjectDetect = new ObjectDetectors(data);
  const results = await ObjectDetect.process();
  res.json(results.data[0].class);
});

server.listen(PORT, () => console.log(`== Listening on port ${PORT}`));

(async () => {
  const model = await cocoSsd.load();

  console.log("== Coco SSD has successfully loaded");
})();
