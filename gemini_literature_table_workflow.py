'''
Adapted from https://ai.google.dev/docs/semantic_retriever
TODO:
    * check if doc already in corpus 
    * Add printing switch to AQA
    * Add temperature argument
'''
import os
# import numpy as np
import pandas as pd
from google.oauth2 import service_account
from langchain_community.document_loaders import PyPDFLoader
import pprint
# from pyzotero import zotero
import google.ai.generativelanguage as glm


def printJSON(dictio):
    pp = pprint.PrettyPrinter(indent=4)
    print(pp.pformat(dictio[0]))

# def to_markdown(text):
#   text = text.replace('•', '  *')
#   return textwrap.indent(text, '> ', predicate=lambda _: True)


def answer_to_markdown(aqa_response):
    '''Prints all parts of response as Markdown.'''
    markdown_content = "\n".join([aqa_response.answer.content.parts[i].text
                                  for i in range(len(aqa_response.answer.content.parts))])
    print(markdown_content)
    return markdown_content


def ingest_doc(pdf_pth, doi):
    ## Ingest Doc w LangChain pdf loader
    '''An advantage of this approach is that documents can be retrieved with page numbers.'''
    # pdf_pth = '/Users/ekyzivat/Zotero/storage/DGLK8D4E/JGR Biogeosciences - 2015 - Tan - Methane emissions from pan‐Arctic lakes during the 21st century  An analysis with.pdf'
    # one doc/page, just like pdfminer, and also includes page number
    loader = PyPDFLoader(pdf_pth)
    pages = loader.load_and_split()

    # Create a document with a custom display name.
    document = glm.Document(display_name=os.path.basename(pdf_pth))

    # Add metadata.
    # Metadata also supports numeric values not specified here
    document_metadata = [
        glm.CustomMetadata(key="doi", string_value=doi)]
    document.custom_metadata.extend(document_metadata)

    # Make the request
    # corpus_resource_name is a variable set in the "Create a corpus" section.
    create_corpus_request = glm.CreateDocumentRequest(
        parent=corpus_resource_name, document=document)
    create_document_response = retriever_service_client.create_document(
        create_corpus_request)

    # Set the `document_resource_name` for subsequent sections.
    document_resource_name = create_document_response.name
    print(f'Created doc: {document_resource_name}')
    return pages, document_resource_name


def get_doc(document_resource_name):
    '''    Use the `GetDocumentRequest` method to programmatically access the document you created above. The value of the `name` parameter refers to the full resource name of the document and is set in the cell above as `document_resource_name`. The expected format is `corpora/corpus-123/documents/document-123`.
    '''
    # doc_res_name = 'corpora/procbasedmodelsv1-qzyc5wpfg8rw/documents/jgr-biogeosciences-2015-tan-t34u705c2i9p'
    get_document_request = glm.GetDocumentRequest(name=document_resource_name)

    # Make the request
    # document_resource_name is a variable set in the "Create a document" section.
    get_document_response = retriever_service_client.get_document(
        get_document_request)

    # Print the response
    print(get_document_response)


def chunk_doc(pages):
    # Create `Chunk` entities.
    chunks = []
    for passage in pages:
        chunk = glm.Chunk(
            data={'string_value': passage.to_json()['kwargs']['page_content']})
        # Optionally, you can add metadata to a chunk
        # chunk.custom_metadata.append(glm.CustomMetadata(key="tags",
        #                                                 string_list_value=glm.StringList(
        #                                                     values=["Google For Developers", "Project IDX", "Blog", "Announcement"])))
        chunk.custom_metadata.append(glm.CustomMetadata(key="My_chunking_strategy",
                                                        string_value="pymupdf"))
        chunks.append(chunk)
    return chunks


def uploadChunks(document_resource_name, chunks):
    ## Batch create chunks and send to API
    create_chunk_requests = []
    for chunk in chunks:
        create_chunk_requests.append(glm.CreateChunkRequest(
            parent=document_resource_name, chunk=chunk))

    # Make the request
    request = glm.BatchCreateChunksRequest(
        parent=document_resource_name, requests=create_chunk_requests)
    response = retriever_service_client.batch_create_chunks(request)
    print(response)


def listUploadedChunks(document_resource_name):
    ### List `Chunk`s and get state
    # Make the request
    request = glm.ListChunksRequest(parent=document_resource_name)
    list_chunks_response = retriever_service_client.list_chunks(request)
    for index, chunks in enumerate(list_chunks_response.chunks):
        print(f'\nChunk # {index + 1}')
        print(f'Resource Name: {chunks.name}')
        # Only ACTIVE chunks can be queried.
        print(f'State: {glm.Chunk.State(chunks.state).name}')


def AQA(user_query, corpus_resource_name, generative_service_client, answer_style="ABSTRACTIVE", doi_filter=None):
    # Or ABSTRACTIVE, VERBOSE, EXTRACTIVE
    MODEL_NAME = "models/aqa"
    # user_query = 'What type of model dide Gao et al. [2013] use to project CH4 emissions from the lakes?'
    # Make the request
    # corpus_resource_name is a variable set in the "Create a corpus" section.
    content = glm.Content(parts=[glm.Part(text=user_query)])
    # Add metadata filters for both chunk and document.
    if doi_filter is not None:
        document_metadata_filter = [glm.MetadataFilter(key='document.custom_metadata.doi',
                                                       conditions=[glm.Condition(
                                                           string_value=doi_filter,
                                                           operation=glm.Condition.Operator.EQUAL)])]
    else:
        document_metadata_filter = None
    retriever_config = glm.SemanticRetrieverConfig(
        source=corpus_resource_name, query=content, metadata_filters=document_metadata_filter)
    req = glm.GenerateAnswerRequest(model=MODEL_NAME,
                                    contents=[content],
                                    semantic_retriever=retriever_config,
                                    temperature=0.2,
                                    answer_style=answer_style)
    aqa_response = generative_service_client.generate_answer(req)
    answer = answer_to_markdown(aqa_response)
    print(
        f"Answerable probability: {aqa_response.answerable_probability:.0%}")
    return answer, aqa_response.answerable_probability


if __name__ == '__main__':

    ## User vars
    output_dir = '/Volumes/metis/GAI/Lit-review-matrices'
    v = 2

    ## GCP vars
    # Set the path to your service account key file.
    # Rename the uploaded file to `service_account_key.json` OR
    # Change the variable `service_account_file_name` in the code below.
    service_account_file_name = '/Users/ekyzivat/.gcp/keys/sturdy-hangar-410923-09c1dbf6a5bb.json'

    credentials = service_account.Credentials.from_service_account_file(
        service_account_file_name)

    scoped_credentials = credentials.with_scopes(
        ['https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/generative-language.retriever'])

    generative_service_client = glm.GenerativeServiceClient(
        credentials=scoped_credentials)
    retriever_service_client = glm.RetrieverServiceClient(
        credentials=scoped_credentials)
    permission_service_client = glm.PermissionServiceClient(
        credentials=scoped_credentials)

    # ## Create a corpus
    # ## Use corpus resource name: corpora/procbasedmodelsv1-qzyc5wpfg8rw
    # example_corpus = glm.Corpus(display_name="proc-based-models-v1")
    # create_corpus_request = glm.CreateCorpusRequest(corpus=example_corpus)

    # # Make the request
    # create_corpus_response = retriever_service_client.create_corpus(create_corpus_request)

    # # Set the `corpus_resource_name` for subsequent sections.
    # corpus_resource_name = create_corpus_response.name
    # print(create_corpus_response)

    ### Get the created corpus
    # run line if corpus a already exists
    corpus_resource_name = 'corpora/procbasedmodelsv1-qzyc5wpfg8rw'
    get_corpus_request = glm.GetCorpusRequest(name=corpus_resource_name)

    # Make the request
    get_corpus_response = retriever_service_client.get_corpus(
        get_corpus_request)

    # Print the response
    print(get_corpus_response)

    ######################
    # Upload files
    ######################
    pdf_list = [
        '/Users/ekyzivat/Zotero/storage/DGLK8D4E/JGR Biogeosciences - 2015 - Tan - Methane emissions from pan‐Arctic lakes during the 21st century  An analysis with.pdf',
        '/Users/ekyzivat/Zotero/storage/457SYDQ9/Schmid et al. - 2017 - Role of gas ebullition in the methane budget of a .pdf',
        '/Users/ekyzivat/Zotero/storage/93TM6PUP/Guo et al. - 2020 - Rising methane emissions from boreal lakes due to .pdf',
        '/Users/ekyzivat/Zotero/storage/CBZG3MAE/Harrison et al. - 2021 - Year-2020 Global Distribution and Pathways of Rese.pdf',
        '/Users/ekyzivat/Zotero/storage/BNHP5D94/Yuan et al. - 2021 - Effect of water-level fluctuations on methane and .pdf',
        '/Users/ekyzivat/Zotero/storage/E3NGDE38/Delwiche et al. - 2022 - Estimating Drivers and Pathways for Hydroelectric .pdf'
    ]
    doi_list = [
        '10.1002/2015JG003184',
        '10.1002/lno.10598',
        '10.1088/1748-9326/ab8254',
        '10.1029/2020GB006888',
        '10.1016/j.jhydrol.2021.126169',
        '10.1029/2022JG006908'
    ]

    # for i, pdf_pth in enumerate(pdf_list):
    #     if None:
    #         pass
    #     pages, document_resource_name = ingest_doc(pdf_list[i], doi_list[i])

    #     get_doc(document_resource_name)

    #     chunks = chunk_doc(pages)

    #     uploadChunks(document_resource_name, chunks)

    #     print(f'uploaded chunked doc: {document_resource_name}')

    ## AQA

    # AQA('Please summarize Delwiche et al.', corpus_resource_name, generative_service_client)
    # AQA('Please summarize Delwiche et al.', corpus_resource_name, generative_service_client, doi_filter='10.1016/j.jhydrol.2021.126169')  # should have low probability
    # AQA('Did this study look at lakes or reservoirs?', corpus_resource_name, generative_service_client,
    #     'EXTRACTIVE', '10.1029/2022JG006908')

    # Define the data rows
    cols = ['Study', 'Model', 'Input',
            'Resolution', 'Domain', 'Dims', 'Type', 'Outputs']
    queries = [
        'Please print the in-text citation for this paper, without using parentheses (e.g. Tan & Zhuang 2015, Environmental Research Letters)',
        'What biogeochemical model was used in the study?',
        'What input data did the model take in?',
        'What was the spatial resolution of the model, or grid cell length, if indicated?',
        'What What is the geographic domain for which the model was run?',
        'Was the model one dimensional, two dimensional, or other?',
        'Was the model used for lakes, reservoirs, or another type of water body?',
        'What were the output variables of the model?'
    ]
    df = pd.DataFrame(index=range(6), columns=cols, dtype=str)
    df_probs = pd.DataFrame(index=range(6), columns=cols)
    for i, doi in enumerate(doi_list):
        print(f'\nReading paper {doi}...')
        for j, query in enumerate(queries):
            print(f' > {query}')
            aqa_answer, aqa_prob = AQA(
                query, corpus_resource_name, generative_service_client, answer_style='EXTRACTIVE', doi_filter=doi)
            if aqa_prob > 0.5:
                df.iloc[i, j] = aqa_answer
            else:
                df.iloc[i, j] = 'N/A'
            df_probs.iloc[i, j] = aqa_prob
    print(df)
    df.to_excel(os.path.join(
        output_dir, f'lit_review_matrix_v{v}.xlsx'), sheet_name='Responses')
    df_probs.to_excel(os.path.join(
        output_dir, f'lit_review_matrix_v{v}.xlsx'), sheet_name='Probabilities')

    # Create an ExcelWriter object
    excel_writer = pd.ExcelWriter(
        os.path.join(output_dir, f'lit_review_matrix_v{v}.xlsx'), engine='openpyxl')

    # Write the first DataFrame to the 'Responses' sheet
    df.to_excel(excel_writer, sheet_name='Responses')

    # Write the second DataFrame to a new sheet named 'Probabilities'
    df_probs.to_excel(excel_writer, sheet_name='Probabilities')

    # Save and close the ExcelWriter
    excel_writer.close()
    pass

    ## SCRAP
    '''Please make a literature review matrix for the following articles. Include columns for in-text citation, name of biogeochemical model, input data, spatial resolution, the spatial domain for where the model was run, whether the model is 1 or 2-dimensional, and outputs (e.g. CH4 emissions, concentration, or other variables). If you can't find the answer, leave that column blank. '''
