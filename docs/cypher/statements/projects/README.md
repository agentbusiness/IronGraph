# Project statements

A project is a named, isolated graph. Every query names one, and nothing falls through to a default — an application cannot accidentally read or write the wrong graph because it forgot to say which.

| Page | Summary | Standard |
| --- | --- | --- |
| [`CREATE PROJECT`](./create-project.md) | Creates a named, isolated graph. | extension |
| [`SHOW PROJECTS`](./show-projects.md) | Lists every project with its stable identity and display name. | extension |
| [`ALTER PROJECT`](./alter-project.md) | Renames a project while preserving its identity and contents. | extension |
| [`DROP PROJECT`](./drop-project.md) | Removes a project and, with `CASCADE`, everything inside it. | extension |
