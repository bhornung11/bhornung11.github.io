---
layout: post
title:  "Airline codes. Part 1.: Data pipeline"
date:   2026-02-05 19:41:14 +0000
categories: async, data pipeline, SQL
---

## Introduction

A doublet of posts is dedicated to uncover the rules by which the two-letter codes are assigned to airlines. To add suspense to the already exceedingly high level of implied entertainment value of these notes, this first post only concerns itself with building a data gathering, extraction and cleaning pipeline.

## Note

The source code of the pipeline has been placed in [this](https://github.com/bhornung11/airline_codes) repository. The data which is processed is located in [this](https://github.com/bhornung11/bhornung11.github.io/tree/main/assets/airline-codes-01/data/html) folder. Links to various utilities and functions are provided throughout the following paragraphs.

## Aim and scope

The final deliverable is a service that proposes two letter (`IATA`-style) codes of names of airlines that either exists of have existed. Each suggested code must have an associated probability. The product of this segment of work is a pipeline that gathers the required data and writes it to a database once it has been cleaned. In terms of data quality analysis, the pipeline only concerns itself with whether the data is of the correct format.

## Data and its source

The codes of the airlines along with other basic pieces of information were obtained from the [avcodes](https://www.avcodes.co.uk/airlcodesearch.asp) website. To protect the environment and the server, a set of `html` pages were downloaded. Each of them enumerates all airlines returned to searching for a single letter or digit. It is therefore expected that entries can appear in multiple files.

### Data format in the `html` documents

The pages contain tables which collate various pieces of information about airlines. These records are expected to be consistently formatted for they are created programatically as a response to a search query. An example of them is provided below to make up for the lack of figures. It also shows what `css` classes and text token will be searching for to extract the details.

```html
<main class="form-result">

<p class='text-warning text-uppercase'>This is Current Data</p>

<table class="table table-striped table-bordered border-dark">
	    <tr>
        <td colspan="3" align="center">Aegean Airlines</td>
      </tr>
      <tr>
        <td colspan="5" align="center"><img src="images/logos/AEE.png"></td>
      </tr>
      <tr>
	      <td colspan="3" align="center">Full Name: Aegean Airlines, S.A.</td>
      </tr>
      <tr>
        <td align="center">IATA Code:<br />&nbsp;A3</td>
        <td align="center">ICAO Code:<br />&nbsp;AEE</td>
        <td align="center">ICAO Callsign:<br />&nbsp;Aegean</td>
      </tr>
      <tr>
        <td align="center">IATA Accounting Code:<br />&nbsp;390</td>
        <td align="center">IATA Prefix Code:<br />&nbsp;390</td>
        <td align="center">Country:<br />&nbsp;Greece</td>
      </tr>
      <tr>
        <td align="center" colspan="2">Website URL<br /><a href="http://www.aegeanairlines.gr" target="airlineurl">www.aegeanairlines.gr</a></td>
        <td align="center"><img src="images/flags/GR.gif"></td>
      </tr>
      <tr>
        <td align="center">Founded:<br />&nbsp;</td>
        <td align="center">Commenced Ops:<br />&nbsp;</td>
        <td align="center">Ceased Ops:<br />&nbsp;</td>
      </tr>
      <tr>
        <td colspan="3" align="center">Remarks: Name changed from Aegean Aviation S.A. Merger with Cronus Airlines X5/CUS&nbsp;</td>
      </tr>
</table>
<font size="1">Last Updated: 08/01/2002</font>
</main>
```

## Implementation

### Usecase

First and foremost, the usecase will determine the implementation. A usecase is the collectivity the amount and environment of the data, the user related by the means by which and end to which they will be interacting with the utility.

* the initial `html` documents are of manageable size (0.27MB on average) and number (36)
    * $\rightarrow$ we can keep all documents and intermediate data in the `RAM` (but we won't).
* the final number of entries will be around $36 \cdot 36 \cdot 2$ if we allow for each airline code appear in two searches
    * $\rightarrow$ expect low number of queries and transformations 
* the pipeline will be run every now and then
    * $\rightarrow$ documentation important
* by the writer of these notes
    * $\rightarrow$ documenting the project has just become ever more crucial
* with the intent of reusing its components
    * $\rightarrow$ think about speed

A tool that is reasonably fast, performs the computations locally, in the `RAM` and which is called in a shell will suffice.

#### Approach and tools

It might seem appealing to use `dvc` or `airflow` to drive the pipeline. We resist the temptation due to the real considerations:
* `dvc` pipelines really are powerful for multi-stage model building when the object are written to the disk We, however, would like to avoid I/O operation on the disk. They are costly.
* `airflow` excels at orchestrating tasks which act on data external to their flow. That is to say, it was not designed to pass data between stages of the pipeline. The `XCom` serialisation both restricts and slows down the data manipulations. The running `airflow` given the amount of data also unnecessarily lengthens the processing time.

Therefore the solution will comprise of a sequence of native `python` functions. The connoisseurs of functional programming will likely to be left wishing more linked generators (or coroutines for data will be split). These days, the scribbler of these lines has a strong appreciation of tracebacks not in chains.

An important note, however is in order. It may have been more conducive to formulate the tasks as database queries from the developer experience (`DX`) point of view. By having done some, the number of types could have been greatly reduced. The machinery of the steps would also be the same. This would, of course, happen on the expense of the visibility on the data flow.

As to the third party packages, the following ones were indeed helpful in this project:

* `selectolax`: a fast parser to process `html` documents
* `duckdb` with `pyarrow` to save, serve, access and query data


## Implementation and discussion

The [pipeline](https://github.com/bhornung11/airline_codes/blob/main/src/pipeline.py) is spread out below in its entirety at first. Its components will then be discussed briefly one-by-one.

```python
async def pipeline(path_config: str) -> None:
    """
    Pipeline to retrieve airline detail html files and
    and extract, clean, save data from them.

    Parameters:
        path_config: str : absolute path to the pipeline config file

    Returns:
        None : saves data to db-s as specified in the config file
    """

    # 0.) read resource db names, schemata
    cfg = read_config(path_config)

    # 1.) collect all document ids from the server
    document_ids: List[str] = get_document_ids(
        cfg["url_ids"], timeout=TIMEOUT
    )

    # 2.) retrieve the html document at each id
    document_retriever_kwargs = [
        {
            "url": cfg["url_document"],
            "document_id": document_id, "timeout": TIMEOUT
        }
        for document_id in document_ids
    ]

    document_retriever = make_async_bounded_generator(
        get_document_at_id, LIMIT_REQUEST
    )

    # this is an async generator, documents are not collected
    # until the tables are being extracted
    documents = document_retriever(document_retriever_kwargs)

    # 3.) extract the tables from each html documents done-by-one
    # by requesting a download and processing it immediately
    # once it has completed
    table_extractor = make_async_consumer(
        collect_airline_tables_from_html
    )

    tables_in_document_all: List[NodesDict] = await table_extractor(
        documents,
        cfg["patterns"]["airline_table"]
    )

    # 4. check whether there were tables in the documents
    assert_table_number_in_documents(
        tables_in_document_all, document_ids,
        cfg["kwargs"]["assert_table_number"]["n_tables_required"]
    )

    # 5. extract the raw fields from each table
    records_raw_all: List[RawRecordsDict] = extract_fields_from_airline_html_tables_all(
        tables_in_document_all, cfg["patterns"]["airline_table_fields"]
    )

    # 6. check whether the extraction succeeded w/o
    # unexpected errors (some errors are anticipated)
    ids_excluded: Set[RecordIndex] = assert_raw_records(
        records_raw_all,
        cfg["kwargs"]["extracted_checker"],
        cfg["kwargs"]["extracted_errors_checker"],
        cfg["name_db"],
        "extracted_errors",
        cfg["schemata"]["extracted_errors"]
    )

    # 7. merge all records to a single collection except those which are defective
    records_raw: RawRecordsDict = merge_raw_records(
        records_raw_all, ids_excluded
    )

    # 8. extract the raw strings from the intermediate dict
    parse_raw_airline_records(
        records_raw,
        cfg["patterns"]["airline_table_fields"],
        cfg["name_db"],
        "parsed",
        "parsed_errors",
        cfg["schemata"]["parsed"],
        cfg["schemata"]["parsed_errors"]
    )

    # 9. check and cast strings according to the type of their respective fields
    clean_airline_table(
        cfg["name_db"],
        "parsed",
        "cleaned",
        cfg["schemata"]["cleaned"]
    )

    # 10. screen for missing of malformatted entries
    collect_clean_airline_table_errors(
        cfg["name_db"],
        "parsed",
        "cleaned",
        "cleaned_errors",
        cfg["schemata"]["cleaned_errors"]
    )
```

### Configuration

The pipeline requires directives as to whence retrieve the data, how to process it and where to write the clean records, intermediate data and erroneous entries. These include path to the document resource, names of the databases, tables therein and their schemata, parser patterns. They are provided in a configuration file. The path to this configuration inventory is supplied to the pipeline through the command line. This enables parametrising the workflow without changing the source code. 

```python
cfg = read_config(path_config)
```

The position and the simple signature of the function `read_config` ([source](https://github.com/bhornung11/airline_codes/blob/66327560871c6c3001e60ae9a994f1105f0cd755/src/utils/util_config.py#L23)) allows for replacing it with an other which fetches the configuration, say, from a database.

### Listing available documents - `get_document_ids`

The `html` texts are handled by a server. Firstly, it is queried what documents are available by the `get_document_ids`  [function](https://github.com/bhornung11/airline_codes/blob/66327560871c6c3001e60ae9a994f1105f0cd755/src/utils/util_retrieval.py#L17):

```python
document_ids = get_document_ids(
    cfg["url_ids"], timeout=TIMEOUT
)
```

This pattern enables the pipeline to respond to the changes to the set of existing `html` source files.

### Retrieving the `html` documents - `get_document_at_id`

Interacting with external data sources, reading files implies waiting time. These can add up slowing down the pipeline if requesting a larger number of documents sequentially. It is therefore ensured that multiple files are asked for simultaneously. The `get_document_at_id(url: str, id: str, timeout: int) ->str` [function](https://github.com/bhornung11/airline_codes/blob/66327560871c6c3001e60ae9a994f1105f0cd755/src/utils/util_retrieval.py#L51) loads a file at the specified id. It is wrapped by the `make_async_bounded_generator` [decorator](https://github.com/bhornung11/airline_codes/blob/66327560871c6c3001e60ae9a994f1105f0cd755/src/utils/util_async.py#L23) turning it to an asynchronous generator.

```python
document_retriever = make_async_bounded_generator(
    get_document_at_id, LIMIT_REQUEST
documents = document_retriever(document_retriever_kwargs)
)
```

The number of parallel request is limited by the aptly named `limit` parameter. This is to be gentle on the server and to ensure that the number of downloads in the `RAM` is bounded.


#### Note on using a document server

As to why a server is used, the aim is to emulate communication with external resources. The [server](https://github.com/bhornung11/airline_codes/blob/66327560871c6c3001e60ae9a994f1105f0cd755/src/resource/html_server.py#L39) itself is a `FastAPI` instance. A  [resource handler](https://github.com/bhornung11/airline_codes/blob/66327560871c6c3001e60ae9a994f1105f0cd755/src/resource/html_resource.py#L19), `HtmlResourceHandler`, is injected in the server object. It runs independently of the pipeline as a data provider would do in a real life scenario.

### Extracting the raw tables - `collect_airline_tables_from_html`

Each `html` document contains a number of tables. These are extracted with the `collect_airline_tables_from_html` [function](https://github.com/bhornung11/airline_codes/blob/66327560871c6c3001e60ae9a994f1105f0cd755/src/utils/util_extraction.py#L25). It parses the text to document tree. It then finds and collects the tables by matching against a `html` tag--`css` class pair that designates to them. Each table is assigned an id: the document id and the order of the table in the text. The tables themselves are `LexborNode` objects. We do not parse the fields from them just yet. Th extractor function is wrapped by `make_async_consumer` ([source](https://github.com/bhornung11/airline_codes/blob/66327560871c6c3001e60ae9a994f1105f0cd755/src/utils/util_async.py#L92)) so that it can consume the `async` generator of documents producing a list of dictionaries `tables_in_document_all`.

```python
table_extractor = make_async_consumer(
    collect_airline_tables_from_html
)

tables_in_document_all = await table_extractor(
    documents, cfg["patterns"]["airline_table"]
)
```

### Checking the number of tables in the documents - `assert_table_number_in_documents`

It might as well happen that the string used to locate the tables is no longer valid. For instance, the `css` class has change since the last processing. We check if there were tables in each document. If not, the pipeline is terminated by throwing an error. It is known from the data exploration that each document contains exactly one empty table and numerous ones with details in them. Therefore we check whether there are at least two elements (tables) in each dictionary per document. This is carried out by the _ad hoc_ `assert_table_number_in_documents` [function](https://github.com/bhornung11/airline_codes/blob/66327560871c6c3001e60ae9a994f1105f0cd755/src/utils/util_qa.py#L34).

### Finding the matching tags in the tables - `extract_fields_from_airline_html_tables`

Once we are satisfied that the tables were indeed found, the tags of interest are extracted by the `extract_fields_from_airline_html_tables_all` [function](https://github.com/bhornung11/airline_codes/blob/66327560871c6c3001e60ae9a994f1105f0cd755/src/utils/util_extraction.py#L77). It matches strings against the `css` tags and text content of the tags in the parsed `LexborNode` tables. The resultant object is a dictionary where each field has an associated list of raw text matches.

### Screening for incorrect raw records - `assert_raw_records`

In an ideal world, each field has one matching raw string in all tables. The one we happen to live in entertain us with deviations of this desideratum. It is known from previous adventures across the hinterland of data what defects are present. It is checked whether only those thus far discovered are present by the `assert_raw_records` [function](https://github.com/bhornung11/airline_codes/blob/66327560871c6c3001e60ae9a994f1105f0cd755/src/utils/util_qa.py#L80). It collects the id-s of all records (`ids_excluded`) which a defective in the permissible way for the subsequent cleaning. These are from tables which contain no airline information.

Should any table be ridden with issues not anticipated, their id-s and the error are written to the `SQL` table called `extracted_errors`. The pipeline is terminated in this scenario.

### Collating the raw records - `merge_raw_records`

The correct raw records are collected in a single dictionary `records_raw` by the `merge_raw_records` helper [function](https://github.com/bhornung11/airline_codes/blob/66327560871c6c3001e60ae9a994f1105f0cd755/src/utils/util_db.py#L147). The entries which are empty are excluded from the merge.

### Clean the raw records - `parse_raw_airline_records`

The items in the `raw_records` dictionary contain the matched `html` elements. The elements also enclose text which is used to identify what datum of an airline is next to it. However, both in themselves do not carry information on the airline. They are thus removed so only actual datum string is retained e.g. a code, a literal date etc... The parsed records are written to the `SQL` table named `parsed`. Should we fail to locate these substrings, the raw strings are written to the `parsed_errors` `SQL` table for debugging. This all is done by the `parse_raw_airline_records` [function](https://github.com/bhornung11/airline_codes/blob/66327560871c6c3001e60ae9a994f1105f0cd755/src/utils/util_cleaning.py#L274).

### Casting the datum literals - `clean_airline_table`

This is a simple step where the literal strings are attempted to be cast according to their expected types. It is carried out by the means of two `SQL` queries on the `parsed` table in `clean_airline_table` ([source](https://github.com/bhornung11/airline_codes/blob/66327560871c6c3001e60ae9a994f1105f0cd755/src/utils/util_cleaning.py#L31)). The successfully converted items are stored in the `cleaned` `SQL` table.

### Collecting casting errors - `collect_clean_airline_table_errors`

The last step in nothing more than making a tally of the casting errors performed by the `collect_clean_airline_table_errors` [step](https://github.com/bhornung11/airline_codes/blob/66327560871c6c3001e60ae9a994f1105f0cd755/src/utils/util_cleaning.py#L178). There are two classes of fields according to their importance
1. `name_airline`, `iata_code`: these must be non-`null` in the `cleaned` table. If they are missing, the pipeline is terminated.
2. all the other fields: an error is signalled if a field is non-`null` in the `parsed` table but happens to be so after casting. This indicates a malformed string.

The former group of errors can be identified by a simple `ISNULL` query on the respective columns of the `cleaned` table. A straightforward comparison query of the columns of the `parsed` and `cleaned` yields the former set of faults.  A list of both them are saved in the `cleaned_errors` table.
