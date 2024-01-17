<!--
 Copyright 2024 IBM Corp.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->
# Contributing

This file provides general guidance for anyone contributing to IBM AutoXAI4Omics. For technical details on improving the code base please see the `DEV_MANUAL.md`, this contains general information on how to contibute to this repositry.

## Branch managment

In this reprositry we aim to work off a three branch structure:

- `dev` : this is where the tool gets actively developed and is class as unstable for production settings. If you wish to contibute by resolving an issue please create a branch based on `dev` and upon completion submitt a pull request to merge back into `dev`
- `test` : this is a staging area where code brought in from `dev` gets offically tested, if successful it will be merged into `main`
- `main` : this is the offical stable code live and where releases will be made from when code is successfully brought in from `test`

Pull requests will require a maintainer approval before being merged.

## Types of Contributions

### Report Bugs

Report bugs via creating an issue.

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Feature Request

If you wish to request a new feature to be added the best way to send feedback is to create an issue

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation/tests

AutoXAI4Omics could always use more documentation, whether as part of the
official docs, in docstrings, or more tests to increase the reliability and coverage.

## Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
   have the right to submit it under the open source license
   indicated in the file; or

(b) The contribution is based upon previous work that, to the best
   of my knowledge, is covered under an appropriate open source
   license and I have the right under that license to submit that
   work with modifications, whether created in whole or in part
   by me, under the same open source license (unless I am
   permitted to submit under a different license), as indicated
   in the file; or

(c) The contribution was provided directly to me by some other
   person who certified (a), (b) or (c) and I have not modified
   it.

(d) I understand and agree that this project and the contribution
   are public and that a record of the contribution (including all
   personal information I submit with it, including my sign-off) is
   maintained indefinitely and may be redistributed consistent with
   this project or the open source license(s) involved.

## License

This project is licensed under the terms of the [Apache License 2.0](LICENSE), and all contributions must also be licensed under these terms. More information can be found at:

<http://www.apache.org/licenses/LICENSE-2.0>

Please include license headers in any newly created files in any contributions.
