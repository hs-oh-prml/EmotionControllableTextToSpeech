# Copyright (c) 2021, Soohwan Kim. All rights reserved.
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
from kospeech.models.las.decoder import DecoderRNN

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

inputs = torch.LongTensor([[1, 1, 2], [3, 4, 2], [7, 2, 0]])
encoder_outputs = torch.rand(3, 100, 32)

decoder = DecoderRNN(num_classes=10, hidden_state_dim=32, max_length=10)
decoder_outputs = decoder(inputs, encoder_outputs, teacher_forcing_ratio=1.0)
print("teacher_forcing_ratio=1.0 PASS")

decoder = DecoderRNN(num_classes=10, hidden_state_dim=32, max_length=10)
decoder_outputs = decoder(inputs, encoder_outputs, teacher_forcing_ratio=0.0)
print("teacher_forcing_ratio=0.0 PASS")
