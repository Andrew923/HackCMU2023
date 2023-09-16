import streamlit as st
from PIL import Image, ImageDraw
import time
import io

with st.sidebar:    
    with st.spinner("Loading..."):
        time.sleep(.5)

    menu = st.sidebar.selectbox(label="", options=["Home", "Introduction", "Our Model", "Team"])

if menu == "Home":
    st.image('images/solar.jpeg')
    st.subheader("Geomagnetic Storm Predictions")
    st.write("""
                Machine Learning Model that uses time series transformers to predict the probability 
                of geomagnetic storms in space with more efficiency than traditional models.
            """)

    
    col1, col2 = st.columns(2)

elif menu == "Introduction":
    st.image("images/space.png")
    st.subheader("A Solar Threat")
    st.write("""
                The Carrington Event of 1859 was a devastating testament to the immense power of our Sun. 
                During this extraordinary solar storm, a colossal solar flare unleashed an electrifying celestial display, 
                casting mesmerizing auroras across the night sky However, beneath this captivating spectacle lurked a harrowing truth: 
                our modern world, so deeply dependent on technology, is perilously susceptible to the whims of 
                space weather. The Carrington Event's relentless assault on telegraph systems, causing equipment 
                malfunctions, electrical surges, and even setting telegraph lines ablaze.
                
                In space, solar flares release intense bursts of energy and charged particles, 
                posing a direct threat to our satellites orbiting above. These high-energy particles can interfere 
                with satellite electronics, disrupt communications, and potentially lead to malfunctions or even permanent damage.
                
                This is a stark warning that we must urgently prioritize the development of advanced predictive 
                tools for geomagnetic storms to safeguard our interconnected, technology-driven 
                society from potential cataclysmic disruptions that threaten to plunge us into darkness.
            """)

elif menu == "Our Model":
    st.image("images/model.png")
    st.subheader("Our Model")
    st.write("""
            We utilized a ML Model that uses time series transformers for predicting the occurance of geomagnetic storms 
            in space with more efficiency than traditional models. Currently the prediction model used by NASA can predict 
            geostorms approximately 30 minutes before they occur, we hope through the utilization of transformer 
            archietecture which addresses the vanishing gradient problems apparent in RNNs, our model will be able to 
            predict the storms up to a day before they occur. 
            
        """)

    

    with st.expander("Data Preprocessing", expanded=False):
        st.write("""
                    The model was trained on data obtained from the National Oceanic and Atmospheric Administration publically available data 
                    from the Deep Space Climate Observatory (DSCOVR) satellite which monitors solar wind from the Lagrange point between the Earth and Sun

                """)
        st.image("images/dscovr.png")

    with st.expander("Transformer Model", expanded=False):
        st.write("""
                Transformer models are also neural networks, but they are better than other neural networks like recurrent neural networks (RNN) and convolutional neural networks (CNN).
                This is because they can process entire input data at once as opposed to processing data sequentially
                Transformer models a mathematical technique called self-attention, 
                to detect subtle ways data elements in a series influence and depend on each other.
                Transformers can detect trends and anomalies to make online recommendations- everytime you use google, you are using a transformer. 
                """)
        st.image("images/1.png")


elif menu == "Team":
    st.snow()
    st.subheader("Meet the Team")
    st.write("")

    derek = Image.open('images/derek.png')
    aditri = Image.open('images/aditri.jpeg')
    andrew = Image.open('images/andrew.png')
    steven = Image.open('images/steven.png')
    
    def draw_circle(image):
        image.resize((100, 100))
        mask = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask)
        width, height = image.size
        radius = min(width, height) // 2
        center = (width // 2, height // 2)
        draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), fill=255)

        # Apply the circular mask to the image
        circular_image = Image.new('RGBA', image.size)
        circular_image.paste(image, mask=mask)
        return circular_image
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(draw_circle(derek), width=300)
        st.text("Derek Lu")
        st.text("Lead Front-End Dev")
        
        st.image(draw_circle(aditri), width=300)
        st.text("Aditri Gupta")
        st.text("Project Manager")
    with col2:
        st.image(draw_circle(andrew), width=300)
        st.text("Andrew Yu")
        st.text("AI Specialist")
        
        st.image(draw_circle(steven), width=300)
        st.text("Steven Guo")
        st.text("Head Researcher")
        


