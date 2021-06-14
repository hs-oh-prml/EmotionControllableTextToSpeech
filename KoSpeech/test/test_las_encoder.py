# Copyright (c) 2020, Soohwan Kim. All rights reserved.
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

import torch
from kospeech.models.las.encoder import EncoderRNN

inputs = torch.rand(3, 12345, 80)
input_lengths = torch.IntTensor([12345, 12300, 12000])

encoder = EncoderRNN(input_dim=80, hidden_state_dim=32, joint_ctc_attention=False, extractor='vgg')
encoder_outputs, encoder_log_probs, encoder_output_lengths = encoder(inputs, input_lengths)
print("joint_ctc_attention=False PASS, VGGExtractor")

encoder = EncoderRNN(input_dim=80, hidden_state_dim=32, joint_ctc_attention=False, extractor='ds2')
encoder_outputs, encoder_output_lengths, encoder_log_probs = encoder(inputs, input_lengths)
print("joint_ctc_attention=False PASS, DeepSpeech2Extractor")

encoder = EncoderRNN(input_dim=80, hidden_state_dim=32, num_classes=2, joint_ctc_attention=True)
encoder_outputs, encoder_output_lengths, encoder_log_probs = encoder(inputs, input_lengths)
print("joint_ctc_attention=True PASS")
