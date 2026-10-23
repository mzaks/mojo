# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""The HAL `DeviceContext` compat layer.

Staged as `max.gpu.host._device_context_hal` so it compiles alongside the
legacy `DeviceContext`, but nothing imports it from here yet. Under
--config=hal-device-context its modules are instead staged over the
`max.gpu.host` modules they replace (see
`max/driver/src/max_license/BUILD.bazel`); this file is not staged there.
"""
