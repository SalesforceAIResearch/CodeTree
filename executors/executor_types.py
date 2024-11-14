# LICENSE HEADER MANAGED BY add-license-header
#
# /*
#  * Copyright (c) 2023, Salesforce, Inc.
#  * SPDX-License-Identifier: Apache-2
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  * you may not use this file except in compliance with the License.
#  * You may obtain a copy of the License at
#  *
#  *     http://www.apache.org/licenses/LICENSE-2.0
#  *
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS,
#  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  * See the License for the specific language governing permissions and
#  * limitations under the License.
#  */
#

from typing import NamedTuple, List, Tuple
from abc import ABC, abstractmethod

class ExecuteResult(NamedTuple):
    is_passing: bool
    feedback: str
    state: Tuple[bool]

class Executor(ABC):
    @abstractmethod
    def execute(self, func: str, tests: List[str], timeout: int = 5) -> ExecuteResult:
        ...

    @abstractmethod
    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
        ...

# class Executor:
#     def execute(self, func: str, tests: List[str], timeout: int = 5) -> ExecuteResult:
#         raise NotImplementedError

#     def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
#         raise NotImplementedError



