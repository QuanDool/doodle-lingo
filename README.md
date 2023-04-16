Welcome to DoodleLingo!

Learning a new language is an integral part of everyone’s education. Learning about cultures vastly different than your own and being able to communicate with a wider variety of people opens the door to new experiences, friendships, career possibilities, travel experiences and more.

But, learning a new language also comes with a lot of memorization. It’s often tedious to learn tens or even hundreds of new words. We wanted to change that.

With DoodleLingo, you can learn a new language by drawing! Incorporating art into education can be fun, while also having a variety of benefits in terms of memory. Have fun with learning, and doodle to your heart's content!

Our project uses a trained CNN using the 7 Keras layers on the quick, draw! Dataset in order to predict what you are drawing. Our web development was done using JavaScript, HTML and CSS with bootstrap. 

The web app is locally hosted and served using the ASGI server Uvicorn in combination with FastAPI with an option to wrap it inside a Docker container.

Just open up the app, and draw the word displayed in both your language and the language you’re learning! Once you get it right, our machine learning model will detect your drawing is complete, and allow you to continue on to the next word. When you see that word again, some letters from the language you’re familiar with will be hidden, so try to remember the word based on just the new language! Need a hint? Click the hint button to reveal more letters of the hidden word. And, if you manage to get the word right when no letters are shown, you’ve mastered a word! Try to master them all!

# Created using:

# quickdraw-cnn

A convolutional neural network using Tensorflow and Google's Quick, Draw! [dataset](https://github.com/googlecreativelab/quickdraw-dataset) to recognize hand drawn images including a webapp to draw them.

Read my [blog post](https://larswaechter.dev/blog/recognizing-hand-drawn-doodles/) for more information. You can find a webapp demo [here](https://quickdraw-cnn.fly.dev/).

![Preview](./webapp.png)

## Setup

### cnn

Switch to the `cnn` directory, create a new virtual environment and install the required packages:

```
python -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```

Then, launch Jupyter in the target directory:

```
jupyter notebook
```

### webapp

#### Native

Switch to the `webapp` directory, create another venv and install the requirements as mentioned above. You can run the webapp using the following command:

```
uvicorn main:app
```

The webapp should be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

#### Docker

Alternatively, you can also run it via Docker:

```
docker build . -t quickdraw-webapp
docker run -p 443:443 quickdraw-webapp
```

The webapp should be available at [http://0.0.0.0:443](http://0.0.0.0:443).
