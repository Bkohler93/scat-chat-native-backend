const tf = require("@tensorflow/tfjs-node");

const cocossd = require("@tensorflow-models/coco-ssd");

const { base64ToUint8Array } = require("base64-u8array-arraybuffer");

function ObjectDetectors(image) {
  this.inputImage = image;

  this.loadCocoSsdModal = async () => {
    const modal = await cocossd.load({
      base: "mobilenet_v2",
    });
    return modal;
  };

  this.getTensor3dObject = (numOfChannels) => {
    const imageData = this.inputImage.image
      .replace("data:image/jpeg;base64", "")
      .replace("data:image/png;base64", "");

    const imageArray = base64ToUint8Array(imageData);

    const tensor3d = tf.node.decodeJpeg(imageArray, numOfChannels);

    return tensor3d;
  };

  this.process = async () => {
    let predictions = null;
    const tensor3D = this.getTensor3dObject(3);
    const model = await this.loadCocoSsdModal();
    predictions = await model.detect(tensor3D);
    if (predictions.length === 0) {
      predictions = [{ class: "none" }];
    }
    tensor3D.dispose();
    return { data: predictions };
  };
}

module.exports = ObjectDetectors;
