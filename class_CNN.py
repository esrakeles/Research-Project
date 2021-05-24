# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.models import load_model

import argparse
import sys
import time

import numpy as np
import tensorflow as tf
import cv2


class OCR():

    def __init__(self, modelFile, labelFile):
        self.model_file = modelFile
        self.label_file = labelFile

        self.labels = self.load_labels(self.label_file)
        self.model = self.load_model(self.model_file)

    def load_model(self, modelFile):
        model = load_model(modelFile)
        return model

    def load_labels(self, labelFile):
        label = []
        proto_as_ascii_lines = tf.io.gfile.GFile(labelFile).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())
        return label

    def read_tensor_from_image(self, image, imageSizeOutput):
        """
        inputs an image and converts to a tensor
        """

        image = cv2.resize(image, dsize=(imageSizeOutput, imageSizeOutput), interpolation=cv2.INTER_CUBIC)
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.array(image, dtype="float32")
        image = tf.expand_dims(image, axis=0)
        return image

    def label_image(self, tensor):
        result = self.model.predict(tensor)
        MaxPosition = np.argmax(result)
        return self.labels[MaxPosition]

    def label_image_list(self, listImages, imageSizeOuput):
        plate = ""
        for img in listImages:
            plate = plate + self.label_image(self.read_tensor_from_image(img, imageSizeOuput))
        return plate, len(plate)