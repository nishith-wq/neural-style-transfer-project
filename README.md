# neural-style-transfer-project

*COMPANY* - *CODTECH IT SOLUTIONS*

*NAME* - *NISHEETH CHANDRA*

*INTERN ID* - *CT04DF2787*

*DOMAIN* - *ARTIFICIAL INTELLIGENCE*

*DURATION* - *4 WEEKS*

*MENTOR* - *NEELA SANTHOSH*

# Neural Style Transfer with TensorFlow and VGG19

## Overview

This project implements a Neural Style Transfer (NST) algorithm using TensorFlow and the VGG19 model. Neural Style Transfer is a technique in deep learning that blends two images—one as the "content" image and the other as the "style" image—to generate a new image that preserves the core content while reflecting the style and textures of the second. It combines principles from computer vision, optimization, and convolutional neural networks to produce visually compelling artworks.

The notebook provides a clean and interpretable implementation of NST, making it a practical educational resource for understanding how convolutional features can be manipulated for creative purposes.

## Objective

The goal is to transform a given content image (e.g., a sea turtle) so that it adopts the artistic style of a famous painting (e.g., "The Great Wave off Kanagawa"). This transformation is achieved by optimizing a noise image to simultaneously minimize content loss (difference from the content image) and style loss (difference from the style image).

## Tools & Libraries

- **TensorFlow 2.x**: For deep learning operations and model management
- **Keras Applications**: For loading the pre-trained VGG19 network
- **PIL & Matplotlib**: For image loading and visualization
- **NumPy**: For numerical processing
- **IPython.display**: To render intermediate results inline

## How It Works

The core architecture relies on a pre-trained **VGG19** convolutional neural network, which is known for its strong feature extraction capability. The model is not trained end-to-end. Instead, its pre-trained weights are used to extract high-level and low-level representations of content and style from the input images.

### Step-by-Step Pipeline

1. **Input Preparation**
   - The content and style images are downloaded, resized to a common resolution, and preprocessed.
   - Images are formatted to the VGG19 input format using `keras.applications.vgg19.preprocess_input`.

2. **Model Selection**
   - VGG19 is loaded with pre-trained ImageNet weights.
   - Only specific intermediate layers are used for style and content extraction.
     - *Content Layer*: Typically `block5_conv2`
     - *Style Layers*: Typically include lower-level convolution layers such as `block1_conv1`, `block2_conv1`, etc.

3. **Loss Functions**
   - **Content Loss**: Mean squared error between feature maps of the content image and generated image.
   - **Style Loss**: Based on the difference between Gram matrices (which capture texture) of the style image and the generated image.
   - **Total Variation Loss** (optional): Encourages spatial smoothness in the generated image.

4. **Optimization**
   - An Adam optimizer is used with customized learning rate and momentum values.
   - The generated image is initialized from the content image and iteratively updated to minimize the combined loss.

5. **Image Generation**
   - After several iterations of forward and backward passes, the generated image increasingly resembles the style image in texture while preserving the layout of the content image.
   - The result is deprocessed (inverse of VGG19 preprocessing) and displayed using matplotlib.

## Results

The generated output is an aesthetically pleasing image that successfully integrates the structure of the content image and the visual characteristics of the style image. With the right balance between style and content loss weights, the transformation achieves a compelling visual blend.

## Applications

- Artistic rendering for digital content creation
- Style augmentation in game design or animation
- Teaching material for convolutional networks and feature extraction
- Research baseline for neural artistic generation models

## Limitations

- High computational requirements (best run on GPU)
- Limited resolution support for large images due to memory constraints
- Style transfer quality is sensitive to layer choice and weight balancing

## Future Enhancements

- Integration with more advanced perceptual loss functions
- Incorporation of GANs for improved stylization
- Real-time style transfer using MobileNet or Fast Style Transfer architectures
- Web-based deployment with Flask or Streamlit for user interaction

## Conclusion

This notebook demonstrates the power of convolutional neural networks to perform complex visual transformations like Neural Style Transfer. It serves as an excellent bridge between artistic creativity and deep learning engineering, showing how AI can be used not just for predictions, but also for visual expression. The implementation is modular, interpretable, and extensible for further experimentation.


