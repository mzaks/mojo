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
"""Staged-tree variant of `_hal/_machine.mojo`.

The `max.gpu.host` overlay stages the `machine` package inside `_hal`, so the
re-exported names resolve relative to it. Staged over `_hal/_machine.mojo` by
`_hal_staged_layer` (see `max/driver/src/max_license/_hal/BUILD.bazel`); never
compiled at this location.
"""

from .machine import MachineDefinition, DeviceRef, DeviceSpec
