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
"""Re-exports the `machine` names `_hal` uses.

Here `machine` is a sibling top-level package; in the staged `max` overlay it
is nested inside `_hal`, and this file is shadowed by a variant importing from
that location (see `_hal_staged_layer` in this package's BUILD.bazel). `_hal` sources import these
names relatively (`from ._machine import ...`), which resolves in both
compilation contexts.
"""

from machine import MachineDefinition, DeviceRef, DeviceSpec
