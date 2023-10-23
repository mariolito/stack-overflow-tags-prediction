# Stack Overflow Tag prediction

This project includes multi-label text classification models, performed on stack-overflow questions.
The analysis includes the following models:
* multi-label classification based on CountVectorizer and TF-IDF features 
* microsoft/codebert-base fine-tuned for multi-label text classification. This would be our main model, which is used in docker application, and provides predictions of test dataset 
at `data/results/codebert`

The aim of this document is to guide you on:
* download all dependencies( libraries, models etc.)
* run applications
* present results

Further reports for analysis and decisions could be found in `stats.ipynb` notebook

## Dependencies


### Install Python Libraries

* Activate virtual enviroment
```angular2
sudo pip install --upgrade virtualenv
mkdir venvs
virtualenv my_venv
source my_venv/bin/activate
```

* Install python libraries
```angular2
pip install -r requirements.txt
```

### Download datasets
* `kaggle datasets download -d stackoverflow/stacksample`
* Extract and copy all data files in `data/`. 


### Download pre-trained models
Run `./running_scripts/downloading/download_hugging_face_models.sh`


## Processes
Run `./running_scripts/preprocessing/preprocess_datasets.sh` to preprocess data

Run `./running_scripts/evaluating/training_baseline_model.sh` to train and evaluate baseline. 
This model could take either tfidf features or count vectorizer, and performs a multi-label classification
either with Logistic or Ridge. Best combinations are for Logistic with Count vectorizer features.

Run `./running_scripts/evaluating/training_bert_model.sh` to train and evaluate baseline


* Validation Set results

|         | Logistic-CountVect | CodeBert |
|---------|---------------------|----------|
| Acc     | 0.49                | 0.75     | 
| F1-macro | 0.65                | 0.75     | 
| F1-micro | 0.66                | 0.84     | 
| AUC-ROC | 0.77                | 0.89     | 


## Applications

The main model will be used in our application:

#### CodeBert multi-label classification
This model is used in docker application. Takes as input the title
and the body of a questions and returns labels and scores.

### Docker Application

Execute:

```angular2
docker build -t tagger .
docker run -p 5000:5000 -d tagger
```

```python
import requests
import json
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
url = 'http://127.0.0.1:5000/predict'
Title = 'Android Camera: app passed NULL surface'
Body ='<p>I\'ve found several questions on this but no answers so here\'s hoping someone might have some insight. When I try to swap the camera I call the swapCamera function below. However the camera preview just freezes (the app is not frozen though just the live camera preview).</p>\n\n<p>When I open the app for the first time everything works just fine. However I noticed something interesting. When I log out the memoryaddress of the _surfaceHolder object (i.e. my SurfaceHolder object) it gives me one value, but whenever I query that value after the app has finished launching and everything, that memory address has changed.</p>\n\n<p>Further still, the error it gives me when I swapCamera is very confusing. I logged out _surfaceHolder before I passed it to the camera in             <code>_camera.setPreviewDisplay(_surfaceHolder);</code>\nand it is NOT null before it\'s passed in.</p>\n\n<p>Any help is greatly appreciated.</p>\n\n<p>I\'ve noticed some interesting behaviour</p>\n\n<pre><code>public class CameraPreview extends SurfaceView implements SurfaceHolder.Callback\n{\n    private SurfaceHolder _surfaceHolder;\n    private Camera _camera;\n    boolean _isBackFacing;\n\n    public CameraPreview(Context context, Camera camera) {\n        super(context);\n        _camera = camera;\n        _isBackFacing = true;\n\n        // Install a SurfaceHolder.Callback so we get notified when the\n        // underlying surface is created and destroyed.\n        _surfaceHolder = getHolder();\n        _surfaceHolder.addCallback(this);\n    }\n\n    void refreshCamera()\n    {\n        try {\n            _camera.setPreviewDisplay(_surfaceHolder);\n            _camera.startPreview();\n        } catch (IOException e) {\n            Log.d("iCamera", "Error setting camera preview: " + e.getMessage());\n        }\n    }\n\n    public void surfaceCreated(SurfaceHolder holder)\n    {\n//        The Surface has been created, now tell the camera where to draw the preview.\n        refreshCamera();\n    }\n\n    public void surfaceDestroyed(SurfaceHolder holder)\n    {\n        // empty. Take care of releasing the Camera preview in your activity.\n        _surfaceHolder.removeCallback(this);\n    }\n\n    public void surfaceChanged(SurfaceHolder holder, int format, int w, int h)\n    {\n         // If your preview can change or rotate, take care of those events here.\n        // Make sure to stop the preview before resizing or reformatting it.\n\n        if (_surfaceHolder.getSurface() == null){\n            // preview surface does not exist\n            return;\n        }\n\n        try {\n            _camera.stopPreview();\n        } catch (Exception e) {\n            // ignore: tried to stop a non-existent preview\n        }\n\n        // set preview size and make any resize, rotate or\n        // reformatting changes her\n        _camera.setDisplayOrientation(90);\n\n        // _startPoint preview with new settings\n        refreshCamera();\n    }\n\n    public void swapCamera()\n    {\n        Camera cam = null;\n        int cameraCount = Camera.getNumberOfCameras();\n        Camera.CameraInfo cameraInfo = new Camera.CameraInfo();\n        _camera.stopPreview();\n        _camera.release();\n        for (int i = 0; i &lt; cameraCount; i++)\n        {\n            Camera.getCameraInfo(i,cameraInfo);\n            if(cameraInfo.facing == Camera.CameraInfo.CAMERA_FACING_FRONT &amp;&amp; _isBackFacing == true)\n            {\n                try\n                {\n                    _camera = Camera.open(i);\n\n                }catch (RuntimeException e)\n                {\n                    Log.e("Error","Camera failed to open: " + e.getLocalizedMessage());\n                }\n            }\n\n            if(cameraInfo.facing == Camera.CameraInfo.CAMERA_FACING_BACK &amp;&amp; _isBackFacing == false)\n            {\n                try\n                {\n                    _camera = Camera.open(i);\n                }catch (RuntimeException e)\n                {\n                    Log.e("Error","Camera failed to open: " + e.getLocalizedMessage());\n                }\n            }\n        }\n\n        _isBackFacing = !_isBackFacing;\n        refreshCamera();\n    }\n}\n</code></pre>\n'
data = {"Body": Body, "Title": Title } 
requests.post(url=url, data=json.dumps(data), headers=headers).json()
```
```json
{
  "labels": ["android"], "scores": ["0.923"]
}                                                                                                                                                                                                                              "0.63",                                                                                                                                                                                                                                      "",                                                                                                                                                                                                                                          "0.77",                                                                                                                                                                                                                                      "",                                                                                                                                                                                                                                          "1.0"                                                                                                                                                                                                                                    ],                                                                                                                                                                                                                                           "sections": [                                                                                                                                                                                                                                    "JOB_INFO",                                                                                                                                                                                                                                  "NONSENSICAL",                                                                                                                                                                                                                               "REQUIREMENTS",                                                                                                                                                                                                                              "NONSENSICAL",                                                                                                                                                                                                                               "BENEFITS"                                                                                                                                                                                                                               ],                                                                                                                                                                                                                                           "sentences": [                                                                                                                                                                                                                                   "This jobs contains tasks for programming. You work close to other developers",                                                                                                                                                              "El candidato o candidata ideal tiene una combinaci\u00f3n \u00fanica de experiencia",                                                                                                                                                       "required to know java. We need someone to join the team of sql",                                                                                                                                                                            "Trabajar en DaCodes te permitir\u00e1 ser vers\u00e1til y \u00e1gil al poder trabajar",                                                                                                                                                     "you have free days of, working remotely, snacks, annual bonus"                                                                                                                                                                          ]                                                                                                                                                                                                                                        }  
```
