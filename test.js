// ==UserScript==
// @name        4chan captcha solver
// @namespace   sneed
// @match       https://boards.4channel.org/*
// @match       https://boards.4chan.org/*
// @match       https://sys.4chan.org/*
// @grant       none
// @version     1.2
// @author      brunohazard; original by AUTOMATIC; slider solver code by HamletDuFromage & soyware
// @require     https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.1/dist/tf.js
// @description 7/8/2021, 1:16:32 PM
// ==/UserScript==
(function() {
  var modelJSON = {"format": "layers-model", "generatedBy": "keras v2.4.0", "convertedBy": "TensorFlow.js Converter v3.7.0", "modelTopology": {"keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, null, 80, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": [null, null, 80, 1], "dtype": "float32", "filters": 40, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "same", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 60, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "same", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": [-1, 1200]}}, {"class_name": "Bidirectional", "config": {"name": "bidi", "trainable": true, "dtype": "float32", "layer": {"class_name": "LSTM", "config": {"name": "lstm", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, "merge_mode": "concat"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 22, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.001, "decay": 0.0, "rho": 0.9, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}, "weightsManifest": [{"paths": ["group1-shard1of1.bin"], "weights": [{"name": "bidi/forward_lstm/lstm_cell_4/kernel", "shape": [1200, 800], "dtype": "float32"}, {"name": "bidi/forward_lstm/lstm_cell_4/recurrent_kernel", "shape": [200, 800], "dtype": "float32"}, {"name": "bidi/forward_lstm/lstm_cell_4/bias", "shape": [800], "dtype": "float32"}, {"name": "bidi/backward_lstm/lstm_cell_5/kernel", "shape": [1200, 800], "dtype": "float32"}, {"name": "bidi/backward_lstm/lstm_cell_5/recurrent_kernel", "shape": [200, 800], "dtype": "float32"}, {"name": "bidi/backward_lstm/lstm_cell_5/bias", "shape": [800], "dtype": "float32"}, {"name": "conv2d/kernel", "shape": [3, 3, 1, 40], "dtype": "float32"}, {"name": "conv2d/bias", "shape": [40], "dtype": "float32"}, {"name": "conv2d_1/kernel", "shape": [3, 3, 40, 60], "dtype": "float32"}, {"name": "conv2d_1/bias", "shape": [60], "dtype": "float32"}, {"name": "dense/kernel", "shape": [400, 22], "dtype": "float32"}, {"name": "dense/bias", "shape": [22], "dtype": "float32"}]}]};
  var charset = ["", "0", "2", "4", "8", "A", "D", "G", "H", "J", "K", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "X", "Y"];
  var model;

  tf.setBackend('cpu'); // takes too long with webgl backend

  /*
   * decide if a pixel is closer to black than to white.
   * return 0 for white, 1 for black
   */
  function pxlBlackOrWhite(r, g, b) {
    return (r + g + b > 384) ? 0 : 1;
  }

  /*
   * Get bordering pixels of transparent areas (the outline of the circles)
   * and return their coordinates with the neighboring color.
   */
  function getBoundries(imgdata) {
    const data = imgdata.data;
    const width = imgdata.width;

    let i = data.length - 1;
    let cl = 0;
    let cr = 0;
    const chkArray = [];
    let opq = true;
    while (i > 0) {
      // alpha channel above 128 is assumed opaque
      const a = data[i] > 128;
      if (a !== opq) {
        if ((data[i - 4] > 128) === opq) {
          // ignore just 1-width areas
          i -= 4;
          continue;
        }
        if (a) {
          /* transparent pixel to its right */
          /*
                  // set to color blue (for debugging)
                  data[i + 4] = 255;
                  data[i + 3] = 255;
                  data[i + 2] = 0;
                  data[i + 1] = 0;
                  */
          const pos = (i + 1) / 4;
          const x = pos % width;
          const y = (pos - x) / width;
          // 1: black, 0: white
          const clr = pxlBlackOrWhite(data[i - 1], data[i - 2], data[i - 3]);
          chkArray.push([x, y, clr]);
          cr += 1;
        } else {
          /* opaque pixel to its right */
          /*
                  // set to color red (for debugging)
                  data[i] = 255;
                  data[i - 1] = 0;
                  data[i - 2] = 0;
                  data[i - 3] = 255;
                  */
          const pos = (i - 3) / 4;
          const x = pos % width;
          const y = (pos - x) / width;
          // 1: black, 0: white
          const clr = pxlBlackOrWhite(data[i + 1], data[i + 2], data[i + 3]);
          chkArray.push([x, y, clr]);
          cl += 1;
        }
        opq = a;
      }
      i -= 4;
    }
    return chkArray;
  }

  /*
   * slide the background image and compare the colors of the border pixels in
   * chkArray, the position with the most matches wins
   * Return in slider-percentage.
   */
  function getBestPos(bgdata, chkArray, slideWidth) {
    const data = bgdata.data;
    const width = bgdata.width;
    let bestSimilarity = 0;
    let bestPos = 0;

    for (let s = 0; s <= slideWidth; s += 1) {
      let similarity = 0;
      const amount = chkArray.length;
      for (let p = 0; p < amount; p += 1) {
        const chk = chkArray[p];
        const x = chk[0] + s;
        const y = chk[1];
        const clr = chk[2];
        const off = (y * width + x) * 4;
        const bgclr = pxlBlackOrWhite(data[off], data[off + 1], data[off + 2]);
        if (bgclr === clr) {
          similarity += 1;
        }
      }
      if (similarity > bestSimilarity) {
        bestSimilarity = similarity;
        bestPos = s;
      }
    }
    return bestPos / slideWidth * 100;
  }

  function getImageDataFromURI(uri) {
    return new Promise((resolve, reject) => {
      const image = document.createElement('img');
      image.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = image.width;
        canvas.height = image.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(image, 0, 0);
        const imgdata = ctx.getImageData(0, 0, canvas.width, canvas.height);
        resolve(imgdata);
      };
      image.onerror = (e) => {
        reject(e);
      };
      image.src = uri;
    });
  }

  /*
   * Automatically slide captcha into place
   * Arguments are the "t-fg', 't-bg', 't-slider' elements of the captcha
   */
  async function slideCaptcha(tfgElement, tbgElement, sliderElement) {
    // get data uris for captcha back- and foreground
    const tbgUri = tbgElement.style.backgroundImage.slice(5, -2);
    const tfgUri = tfgElement.style.backgroundImage.slice(5, -2);

    // load foreground (image with holes)
    const igd = await getImageDataFromURI(tfgUri);
    // get array with pixels of foreground
    // that we compare to background
    const chkArray = getBoundries(igd);
    // load background (image that gets slid)
    const sigd = await getImageDataFromURI(tbgUri);
    const slideWidth = sigd.width - igd.width;
    // slide, compare and get best matching position
    const sliderPos = getBestPos(sigd, chkArray, slideWidth);
    // slide in the UI
    sliderElement.value = sliderPos;
    sliderElement.dispatchEvent(new Event('input'), { bubbles: true });
    return 0 - (sliderPos / 2);
  }

  function toggle(obj,v){
    if(v) obj.style.display = '';
    else obj.style.display = 'none';
  }

  function base64ToArray(base64) {
    var binary_string = window.atob(base64);
    var len = binary_string.length;
    var bytes = new Uint8Array(len);
    for (var i = 0; i < len; i++) {
        bytes[i] = binary_string.charCodeAt(i);
    }
    return bytes.buffer;
  }

  var iohander = {
    load: function(){
      return new Promise((resolve, reject) => {
        resolve({
          modelTopology: modelJSON.modelTopology,
          weightSpecs: modelJSON.weightsManifest[0]["weights"],
          weightData: base64ToArray(weights64),
          format: modelJSON.format,
          generatedBy: modelJSON.generatedBy,
          convertedBy: modelJSON.convertedBy
        });
      });
    }
  }

  async function load(){
    const uploadJSONInput = document.getElementById('upload-json');
    const uploadWeightsInput = document.getElementById('upload-weights-1');
    model = await tf.loadLayersModel(iohander);
    return model;
  }

  // returns ImageData from captcha's background image, foreground image, and offset (ranging from 0 to -50)
  async function imageFromCanvas(img, bg, off){
    var h=img.height, w=img.width;
    var th=80;
    var ph=0, pw=16;
    var scale = th/h

    var canvas = document.createElement('canvas');
    canvas.height = w * scale + pw*2;
    canvas.width = th;

    var ctx = canvas.getContext('2d');

    ctx.fillStyle = "rgb(238,238,238)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.scale(-scale,scale)
    ctx.rotate(90*Math.PI/180);

    var draw = function(off){
      if(bg){
        var border = 4;
        ctx.drawImage( bg, -off + border, 0, w-border*2, h, -w / 2 + border, -h / 2, w-border*2, h );
      }
      ctx.drawImage( img, -w / 2, -h / 2, w, h );
    }

    if(bg && off==null){
      off = await slideCaptcha(document.getElementById('t-fg'), document.getElementById('t-bg'), document.getElementById('t-slider'));
    }
    console.log(off);
    draw(off);

    return ctx.getImageData(0,0,canvas.width,canvas.height);
  }

  // for debugging purposes
  function imagedataToImage(imagedata) {
    var canvas = document.createElement('canvas');
    var ctx = canvas.getContext('2d');
    canvas.width = imagedata.width;
    canvas.height = imagedata.height;
    ctx.putImageData(imagedata, 0, 0);

    var image = new Image();
    image.src = canvas.toDataURL();
    return image;
  }

  async function predict(img, bg, off){
    if(! model){
      model = await load()
    }
    image = await imageFromCanvas(img, bg, off)
    tensor = tf.browser.fromPixels(image, 1).mul(-1/238).add(1)
    prediction = await model.predict(tensor.expandDims(0)).data()

    return createSequence(prediction)
  }

  function createSequence(prediction){
    var csl = charset.length
    sequence = []

    for(var pos=0;pos<prediction.length;pos+=csl){
      var preds = prediction.slice(pos, pos+csl)
      var max = Math.max(...preds);

      var seqElem = {}

      for(var i=0;i<csl;i++){
        var p = preds[i] / max
        var c = charset[i+1]

        if(p>=0.05){
          seqElem[c||""] = p;
        }
      }

      sequence.push(seqElem)
    }

    return sequence
  }

  function postprocess(sequence, overrides){
    csl = charset.length
    possibilities = [{sequence: []}]

    sequence.forEach(function(e, i){
      var additions;
      if(overrides && overrides[i]!==undefined){
        additions = [{sym: overrides[i], off: i, conf: 1}]
      } else{
        additions = Object.keys(e).map(function(sym){ return {sym: sym, off: i, conf: e[sym]} });
      }

      if(additions.length==1 && additions[0].sym=='') return;

      oldpos = possibilities
      possibilities = []
      oldpos.forEach(function(possibility){
        additions.forEach(function(a){
          var seq = [...possibility.sequence]
          if(a.sym != '') seq.push([a.sym, a.off, a.conf])

          var obj = {
            sequence: seq,
          }

          possibilities.push(obj)
        });
      });
    });

    var res = {}
    possibilities.forEach(function(p){
      var line = '';
      var lastSym = undefined;
      var lastOff = -1;
      var count = 0;
      var prob = 0;

      p.sequence.forEach(function(e){
        var sym=e[0];
        var off=e[1];
        var conf=e[2];

        if(sym==lastSym && lastOff + 2 >= off){
          return;
        }

        line += sym;

        lastSym = sym;
        lastOff = off;
        prob += conf;
        count++;
      });

      if(count>0) prob /= count;

      if(prob > res[line] || !res[line]){
        res[line] = prob;
      }
    });

    var keys = Object.keys(res).sort(function(a,b){ return res[a] < res[b]; })
    var keys_fitting = keys.filter(function(x){ return x.length==5 || x.length==6 })
    if(keys_fitting.length>0) keys = keys_fitting;

    return keys.map(function(x){ return {seq: x, prob: res[x]} })
  }



  async function imageFromUri(uri){
    if(uri.startsWith("url(\"")){
      uri = uri.substr(5,uri.length-7)
    }
    if(! uri.startsWith("data:")){
      return null
    }

    var img = new Image();
    await new Promise(r => img.onload=r, img.src=uri);

    return img
  }


  async function predictUri(uri, uribg, bgoff){
    var img = await imageFromUri(uri)
    var bg = uribg ? await imageFromUri(uribg) : null
    var off = bgoff ? parseInt(bgoff) : null;

    return await predict(img, bg, off)
  }


  var solveButton = document.createElement('input')
  solveButton.id="t-auto-solve"
  solveButton.value="Solve"
  solveButton.type="button"
  solveButton.style.fontSize = "11px"
  solveButton.style.padding = "0 2px"
  solveButton.style.margin = "0px 0px 0px 6px"
  solveButton.style.height = "18px"
  solveButton.onclick=async function(){ solve(true); }

  var altsDiv = document.createElement('div')
  altsDiv.id="t-auto-options"
  altsDiv.style.margin = "0"
  altsDiv.style.padding = "0"

  var storedPalceholder;

  var overrides = {}

  function placeAfter(elem, sibling){
    if(elem.parentElement!=sibling.parentElement){
      setTimeout(function(){
        sibling.parentElement.insertBefore(elem,sibling.nextElementSibling);
      }, 1);
    }
  }

  var previousText = null;
  async function solve(force){
    var resp = document.getElementById('t-resp');
    if(! resp) return;

    var bg=document.getElementById('t-bg');
    if(! bg) return;

    var fg=document.getElementById('t-fg');
    if(! fg) return;

    var help=document.getElementById('t-help');
    if(! help) return;

    placeAfter(solveButton, resp)
    placeAfter(altsDiv, help)

    // palememe
    setTimeout(function(){
      toggle(solveButton, bg.style.backgroundImage)
    }, 1);

    var text=fg.style.backgroundImage;
    if(! text){
      altsDiv.innerHTML = '';
      return;
    }

    if(text==previousText && !force) return;
    previousText=text;

    altsDiv.innerHTML = '';
    if(! storedPalceholder) storedPalceholder = resp.placeholder;
    resp.placeholder = "solving captcha...";

    overrides = {}

    var sequence = await predictUri(text, bg.style.backgroundImage, force ? bg.style.backgroundPositionX : null);
    var opts = postprocess(sequence);
    resp.placeholder = storedPalceholder;

    showOpts(opts);
  }

  function showOpts(opts){
    var resp = document.getElementById('t-resp');
    if(! resp) return;

    altsDiv.innerHTML = '';

    if(opts.length == 0){
      resp.value = '';
      return;
    }

    resp.value = opts[0].seq;
  }



  var observer = new MutationObserver(async function(mutationsList, observer) {
    solve(false);
  });



  observer.observe(document.body, { attributes: true, childList: true, subtree: true });

  if(navigator.userAgent.toLowerCase().indexOf('firefox') !== -1){ // request canvas permission on firefox
    let image = new Image(16, 16);
    imageFromCanvas(image, image, 0);
    delete(image);
  }
})();
