import streamlit as st
from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from pickle import load
from numpy import argmax
import tempfile
import os
import streamlit_authenticator as stauth

def main():
    
    usernames = {
        'admin': {
            'name': 'Administrator',
            'password': 'Password',
            'email': 'admin@gmail.com',
        }
    }

    authenticator = stauth.Authenticate({'usernames':usernames}, "caption_generator", "abcdef", cookie_expiry_days=30)
    name, authentication_status, username = authenticator.login("main")

    if authentication_status == False:
        st.error("Username/password is incorrect")

    if authentication_status == None:
        st.warning("Please enter your username and password")

    if authentication_status:
        # Load the pretrained model
        trained_model = load_model('./model_0.h5')
        tokenizer = load(open('./tokenizer.pkl', 'rb'))
        # map an integer to a word
        def word_for_id(integer, tokenizer):
            for word, index in tokenizer.word_index.items():
                if index == integer:
                    return word
            return None

        # extract features from each photo in the directory
        def extract_features(filename):
            vgg16_model = VGG16()
            # re-structure the model
            model = Model(inputs=vgg16_model.inputs, outputs=vgg16_model.layers[-2].output)
            # load the photo
            image = load_img(filename, target_size=(224, 224))
            # convert the image pixels to a numpy array
            image = img_to_array(image)
            # reshape data for the model
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            # prepare the image for the VGG model
            image = preprocess_input(image)
            # get features
            feature = model.predict(image, verbose=0)
            return feature

        # generate a description for an image
        def generate_desc(model, tokenizer, photo, max_length):
            # seed the generation process
            in_text = 'startseq'
            # iterate over the whole length of the sequence
            for i in range(max_length):
                # integer encode input sequence
                sequence = tokenizer.texts_to_sequences([in_text])[0]
                # pad input
                sequence = pad_sequences([sequence], maxlen=max_length)
                # predict next word
                yhat = model.predict([photo,sequence], verbose=0)
                # convert probability to integer
                yhat = argmax(yhat)
                # map integer to word
                word = word_for_id(yhat, tokenizer)
                # stop if we cannot map the word
                if word is None:
                    break
                # append as input for generating the next word
                in_text += ' ' + word
                # stop if we predict the end of the sequence
                if word == 'endseq':
                    break
            return in_text

         # ---- SIDEBAR ----
       
        # Define the header component
        new_title = '<p style="font-size: 40px;">Image Caption Generator App 🔍</p>'
        st.markdown(new_title, unsafe_allow_html=True)

        read_me = st.markdown("""
            Welcome to this Image Caption Generator powered by deep learning! The application utilizes a combination of cutting-edge technologies to provide descriptive captions for your uploaded images. 
            
            Go right ahead and explore meaningful captions to enhance your understanding and enjoyment of images.
        """
        )

        st.sidebar.title(f"Welcome {name}")
        choice = st.sidebar.selectbox("MODE",("Image Captioning","About"))
        authenticator.logout("Logout", "sidebar")

        if choice == "Image Captioning":
            read_me.empty()
            # Hide the disclaimer initially
            st.write("")
            st.write("")

            # Upload image through Streamlit
            uploaded_image = st.file_uploader("Choose an image ...", type=["jpg", "jpeg", "png"])

            if uploaded_image is not None:
                # Display the uploaded image
                st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)
                # Process the image (example: get image dimensions)
                image = Image.open(uploaded_image)
                captioning_btn = st.button("Generate Caption")

                if captioning_btn:
                    sbmt_button = st.success("Generating Caption...")
                    # Create a temporary directory to save the uploaded image
                    temp_dir = tempfile.TemporaryDirectory()
                    temp_file_path = os.path.join(temp_dir.name, "uploaded_image.jpg")
                    # Save the uploaded image to the temporary file
                    image.save(temp_file_path)
                    # extract the features from the image
                    features = extract_features(temp_file_path)
                    # Close and delete the temporary directory
                    temp_dir.cleanup()
                    # generate description
                    caption = generate_desc(trained_model, tokenizer, features, 34)
                    # Clean Up the caption
                    cleaned_caption = caption.replace("startseq", "").replace("endseq", "")
                    sbmt_button.empty()
                    
                    st.write(f"Image Caption: {cleaned_caption}")

if __name__ == "__main__":
    main()
    