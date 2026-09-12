# `SHOW PROJECTS`

> Lists every project with its stable identity and display name.

| | |
| --- | --- |
| Kind | Statement |
| Signature | `SHOW PROJECTS` |
| Relationship to standard Cypher | IronGraph extension |
| Reference dataset | [`trust`](../../datasets.md#trust) — Bitcoin OTC signed trust ratings |

## What it does

`SHOW PROJECTS` returns one row per project: `project_id`, the stable identity the database uses, and `display_name`, the name `USE` matches. It is the one statement that runs without naming a project, because it is about the set of them.

## How it behaves

The identity survives a rename and the display name does not, so anything that needs to refer to a project across time should hold the identity.

The statement stands alone. It cannot be followed by `YIELD`, `WITH` or `WHERE`, and cannot be combined with another statement in the same execution, so filtering and ordering the listing happens in the client.

## When to use it

Use it to discover what exists on a node, to confirm a create or a rename landed, and to resolve a display name to an identity.

## How it differs from its neighbours

`SHOW INDEXES` and `SHOW CONSTRAINTS` describe one project's schema and require a `USE`. `SHOW PROJECTS` describes the node.

## Simple example

Every project on this node, in name order.

```cypher
SHOW PROJECTS
```

Result:

```
project_id                           | display_name        
-------------------------------------+---------------------
03f84ca7-95ca-8972-86c5-3e469d1212c3 | trust               
082b08ee-0417-8733-bfe9-0b5ec717fd14 | citations           
118c3527-7ece-8c52-9858-6dbf3aff394c | temporal_example    
193c9cf8-2fac-889b-8233-86808c5fe1f4 | epinions            
1e4a00e0-579d-8710-9df8-d0c440d4da16 | flights             
2f51a2b8-dcc9-850f-9e35-49e8101cff68 | dblp                
4ef9ade5-83b3-81ca-b145-0d0a29e137c0 | library             
6270cbb3-ff41-818b-ab52-3bd4955d4d0f | overflow            
631b2546-7cd6-84c3-bd8f-71b7d8880032 | email               
ac3c53a6-4508-884b-a884-eb082a9b40da | rollup_example      
b85c6f6f-17a9-8359-b52f-6016e5a180d9 | worked_example_other
bb146a2a-751a-8ccf-b9d1-2fa67a6887f0 | worked_example      
dcad6678-1a4b-8614-ad9a-8c648265a249 | social              
e91bf4cf-df61-83c6-b745-8bed5fc0d196 | backfill_example    

14 rows
```

## Advanced example

The identity is stable and the display name is not. This listing is taken after a rename: the project appears under its new name, carrying the identity it was created with.

```cypher
SHOW PROJECTS
```

Result:

```
project_id                           | display_name          
-------------------------------------+-----------------------
03f84ca7-95ca-8972-86c5-3e469d1212c3 | trust                 
06fd99ec-881d-83fa-aeb3-d9009224ae42 | naming_example_renamed
082b08ee-0417-8733-bfe9-0b5ec717fd14 | citations             
118c3527-7ece-8c52-9858-6dbf3aff394c | temporal_example      
193c9cf8-2fac-889b-8233-86808c5fe1f4 | epinions              
1e4a00e0-579d-8710-9df8-d0c440d4da16 | flights               
2f51a2b8-dcc9-850f-9e35-49e8101cff68 | dblp                  
4ef9ade5-83b3-81ca-b145-0d0a29e137c0 | library               
6270cbb3-ff41-818b-ab52-3bd4955d4d0f | overflow              
631b2546-7cd6-84c3-bd8f-71b7d8880032 | email                 
ac3c53a6-4508-884b-a884-eb082a9b40da | rollup_example        
b85c6f6f-17a9-8359-b52f-6016e5a180d9 | worked_example_other  
bb146a2a-751a-8ccf-b9d1-2fa67a6887f0 | worked_example        
dcad6678-1a4b-8614-ad9a-8c648265a249 | social                
e91bf4cf-df61-83c6-b745-8bed5fc0d196 | backfill_example      

15 rows
```

## Where it earns its place

- Discovering what a node holds.
- Confirming a create or rename.
- Resolving a display name to a stable identity.

## Limitations and trade-offs

- Names and identities only; it reports nothing about size or contents.
- Every project is listed, with no filtering by permission at this layer.

## See also

- [`CREATE PROJECT`](./create-project.md)
