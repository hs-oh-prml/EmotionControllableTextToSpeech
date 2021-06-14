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
from kospeech.models import DeepSpeech2

batch_size = 3
sequence_length = 14321
dimension = 80

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

inputs = torch.rand(batch_size, sequence_length, dimension).to(device)  # BxTxD
input_lengths = torch.LongTensor([14321, 14300, 13000]).to(device)

print("Deep Speech 2 Model Test..")
model = DeepSpeech2(num_classes=10, input_dim=dimension).to(device)
output = model.recognize(inputs, input_lengths)

print(output)
print(output.size())
