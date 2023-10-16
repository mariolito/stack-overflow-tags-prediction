import os
import sys
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
from scipy.special import softmax
import json
from bs4 import BeautifulSoup
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s  - %(message)s', level=logging.INFO)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
device = 'cpu'


def read_models(model_path):
    global classifier, tokenizer, id2label, label2id
    with open(os.path.join(model_path, 'config.json')) as config_f:
        config = dict(json.load(config_f))

    label2id = config['label2id']
    id2label = {j: i for i, j in label2id.items()}

    classifier = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(os.path.dirname(__file__), "..", "data", "models", "codebert"),
        local_files_only=True, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )
    classifier.load_state_dict(
        torch.load(
            os.path.join(
                model_path, config['model_filename']
            )
        )
    )
    classifier.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(os.path.dirname(__file__), "..", "data", "models", "codebert"),
        local_files_only=True
    )
    tokenizer.model_max_length = config['model_max_length']


def _parse_html(html):
    soup = BeautifulSoup(html, features="html.parser")
    # remove all code parts
    text_with_code = soup.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text_with_code.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text_with_code = ' '.join(chunk for chunk in chunks if chunk)
    return text_with_code


def predict(Title, Body):
    Body = _parse_html(Body)
    text = Title + " " + Body
    predicted_label_names, scores = predict_job_sections(text)
    return predicted_label_names, scores


if __name__ == "__main__":
    model_results_path = os.path.join("..", "data", "results", "codebert_model_results")

    read_models(model_results_path)
    print(predict(
        Title='Android Camera: app passed NULL surface',
        Body='<p>I\'ve found several questions on this but no answers so here\'s hoping someone might have some insight. When I try to swap the camera I call the swapCamera function below. However the camera preview just freezes (the app is not frozen though just the live camera preview).</p>\n\n<p>When I open the app for the first time everything works just fine. However I noticed something interesting. When I log out the memoryaddress of the _surfaceHolder object (i.e. my SurfaceHolder object) it gives me one value, but whenever I query that value after the app has finished launching and everything, that memory address has changed.</p>\n\n<p>Further still, the error it gives me when I swapCamera is very confusing. I logged out _surfaceHolder before I passed it to the camera in             <code>_camera.setPreviewDisplay(_surfaceHolder);</code>\nand it is NOT null before it\'s passed in.</p>\n\n<p>Any help is greatly appreciated.</p>\n\n<p>I\'ve noticed some interesting behaviour</p>\n\n<pre><code>public class CameraPreview extends SurfaceView implements SurfaceHolder.Callback\n{\n    private SurfaceHolder _surfaceHolder;\n    private Camera _camera;\n    boolean _isBackFacing;\n\n    public CameraPreview(Context context, Camera camera) {\n        super(context);\n        _camera = camera;\n        _isBackFacing = true;\n\n        // Install a SurfaceHolder.Callback so we get notified when the\n        // underlying surface is created and destroyed.\n        _surfaceHolder = getHolder();\n        _surfaceHolder.addCallback(this);\n    }\n\n    void refreshCamera()\n    {\n        try {\n            _camera.setPreviewDisplay(_surfaceHolder);\n            _camera.startPreview();\n        } catch (IOException e) {\n            Log.d("iCamera", "Error setting camera preview: " + e.getMessage());\n        }\n    }\n\n    public void surfaceCreated(SurfaceHolder holder)\n    {\n//        The Surface has been created, now tell the camera where to draw the preview.\n        refreshCamera();\n    }\n\n    public void surfaceDestroyed(SurfaceHolder holder)\n    {\n        // empty. Take care of releasing the Camera preview in your activity.\n        _surfaceHolder.removeCallback(this);\n    }\n\n    public void surfaceChanged(SurfaceHolder holder, int format, int w, int h)\n    {\n         // If your preview can change or rotate, take care of those events here.\n        // Make sure to stop the preview before resizing or reformatting it.\n\n        if (_surfaceHolder.getSurface() == null){\n            // preview surface does not exist\n            return;\n        }\n\n        try {\n            _camera.stopPreview();\n        } catch (Exception e) {\n            // ignore: tried to stop a non-existent preview\n        }\n\n        // set preview size and make any resize, rotate or\n        // reformatting changes her\n        _camera.setDisplayOrientation(90);\n\n        // _startPoint preview with new settings\n        refreshCamera();\n    }\n\n    public void swapCamera()\n    {\n        Camera cam = null;\n        int cameraCount = Camera.getNumberOfCameras();\n        Camera.CameraInfo cameraInfo = new Camera.CameraInfo();\n        _camera.stopPreview();\n        _camera.release();\n        for (int i = 0; i &lt; cameraCount; i++)\n        {\n            Camera.getCameraInfo(i,cameraInfo);\n            if(cameraInfo.facing == Camera.CameraInfo.CAMERA_FACING_FRONT &amp;&amp; _isBackFacing == true)\n            {\n                try\n                {\n                    _camera = Camera.open(i);\n\n                }catch (RuntimeException e)\n                {\n                    Log.e("Error","Camera failed to open: " + e.getLocalizedMessage());\n                }\n            }\n\n            if(cameraInfo.facing == Camera.CameraInfo.CAMERA_FACING_BACK &amp;&amp; _isBackFacing == false)\n            {\n                try\n                {\n                    _camera = Camera.open(i);\n                }catch (RuntimeException e)\n                {\n                    Log.e("Error","Camera failed to open: " + e.getLocalizedMessage());\n                }\n            }\n        }\n\n        _isBackFacing = !_isBackFacing;\n        refreshCamera();\n    }\n}\n</code></pre>\n'
    ))