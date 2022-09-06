module Page.CosineSimilarity exposing (Data, Model, Msg, page)

import Array
import DataSource exposing (DataSource)
import DataSource.Port as Dport
import Head
import Head.Seo as Seo
import Html exposing (..)
import Html.Attributes exposing (..)
import Html.Parser as HtmlParser exposing (Node(..))
import Html.Parser.Util exposing (toVirtualDom)
import Json.Encode as Encode
import OptimizedDecoder as Decode
import Page exposing (Page, PageWithState, StaticPayload)
import Pages.PageUrl exposing (PageUrl)
import Pages.Url
import Shared
import View exposing (View)


type alias Model =
    ()


type alias Msg =
    Never


type alias RouteParams =
    {}


page : Page RouteParams Data
page =
    Page.single
        { head = head
        , data = data
        }
        |> Page.buildNoState { view = view }


type alias Data =
    List String


data : DataSource Data
data =
    DataSource.combine
        [ Dport.get "parse_katex"
            (Encode.string """cos(\\theta) = \\frac{x y}{\\Vert x \\Vert \\cdot \\Vert y \\Vert}""")
            Decode.string
        , Dport.get "parse_katex"
            (Encode.string """A = \\begin{bmatrix}
a & b \\\\
c & d
\\end{bmatrix}""")
            Decode.string
        , Dport.get "parse_katex" (Encode.string """
AA^T = \\begin{bmatrix}
a & b \\\\
c & d
\\end{bmatrix} \\cdot \\begin{bmatrix}
a & c \\\\
b & d
\\end{bmatrix} =  \\begin{bmatrix}
a^2+b^2 & ac+bd \\\\
ca+db & c^2+d^2
\\end{bmatrix} = \\begin{bmatrix}
x x & xy \\\\
yx & yy
\\end{bmatrix}""") Decode.string
        , Dport.get "parse_katex" (Encode.string """A^2 = \\begin{bmatrix}
a & b \\\\
c & d
\\end{bmatrix}^{ \\circ 2} = \\begin{bmatrix}
a^2 & b^2 \\\\
c^2 & d^2
\\end{bmatrix}""") Decode.string
        , Dport.get "parse_katex" (Encode.string """b = \\begin{bmatrix}
\\sqrt{a^2 + b^2} \\\\
\\sqrt{c^2 + d^2}
\\end{bmatrix} = \\begin{bmatrix}
|x| \\\\
|y|
\\end{bmatrix}""") Decode.string
        , Dport.get "parse_katex" (Encode.string """B = \\begin{bmatrix}
|x| \\\\
|y|\\end{bmatrix} \\cdot \\begin{bmatrix}
|x| &
|y|
\\end{bmatrix} = \\begin{bmatrix}
|x|^2 & |y|\\cdot |x| \\\\
|x|\\cdot |y| & |y|^2
\\end{bmatrix}""") Decode.string
        , Dport.get "parse_katex" (Encode.string """S = A \\circledast \\frac{1}{B}""") Decode.string
        ]


head :
    StaticPayload Data RouteParams
    -> List Head.Tag
head static =
    Seo.summary
        { canonicalUrlOverride = Nothing
        , siteName = "elm-pages"
        , image =
            { url = Pages.Url.external "TODO"
            , alt = "elm-pages logo"
            , dimensions = Nothing
            , mimeType = Nothing
            }
        , description = "TODO"
        , locale = Nothing
        , title = "Cosine Similarity"
        }
        |> Seo.website


view :
    Maybe PageUrl
    -> Shared.Model
    -> StaticPayload Data RouteParams
    -> View Msg
view maybeUrl sharedModel static =
    let
        formulas =
            List.map
                (\formula ->
                    case HtmlParser.run formula of
                        Ok html_formula ->
                            toVirtualDom html_formula

                        Err _ ->
                            toVirtualDom [ Text "error parsing formula " ]
                )
                static.data
    in
    { title = "Cosine Similarity"
    , body = [ body formulas ]
    }


body : List (List (Html Msg)) -> Html Msg
body formulas =
    let
        from_arr =
            Array.fromList formulas
    in
    div [ class "mx-5 sm:mx-0 sm:mx-auto prose lg:prose-lg sm:max-w-xl lg:max-w-3xl mt-10" ]
        [ section []
            [ h2 [] [ text "What is Cosine Similarity?" ]
            , p [] [ text "Cosine similarity is a mathematical fuction that aims to measure the angle between two vecotrs. The angle is an indicator of how close the vectors are to each other. If the angle is 1, then the two vectors lie on top of each other; thus, are exactly the same. If the angle is 0, then the vectors are perpendicular. This is very useful when we want to compare one thing to another and see if they are the same. All we have to do is get a vector representaion of the two objects and apply a cosine similarity function to it. For example, we can compare sentences." ] -- idea: do this blog but with images as well
            , ul []
                [ li [] [ text "\"The quick brown fox jumps over the lazy dogs.\"" ]
                , li [] [ text "\"The quick fox jumped over the dogs.\"" ]
                ]
            , p [] [ text "Semantically, these two sentences are saying the same thing, but the adjectives in the second sentence are removed. Programmatically generating a rule based system to verify if these two sentences are similar would be difficult. In vector form the task is much simpler. We would like to have a vector representation of these two sentences such that their similarity score is close to 1 (i.e. 100%)." ]
            , p [] [ text "Other applications include, impage comparison, plagarism detection, etc." ]
            , section [ class "space-y-5" ]
                [ h2 [] [ text "How do you calculate it?" ]
                , p [] [ text "Below is the formula to calculate cosine similarity" ]
                , case Array.get 0 from_arr of
                    Just val ->
                        div [] val

                    Nothing ->
                        div [] [ text "" ]
                , p []
                    [ text "We perform a dot product on the two vectors and then we normalize with respect to both vectors." ]
                , text "Let x = [1,2,3] and y=[4,5,6]."
                , text "The dot product is [1,2,3] \\cdot [4,5,6] = 3 + 10 + 18 = 32. And the magnitude is 1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14. 4^2 + 5^2 + 6^2 = 77."
                , p []
                    [ text "The similarity between the two vectors is"
                    , text "\\frac{32}{14\\cdot 77} = 0.029."
                    ]
                ]
            , section [ class "space-y-2" ]
                [ h2 [] [ text "A Complex Example" ]
                , p [] [ text "We will use a pretrained deep learning model to generate vector representations of sentenes in an attempt to find similarities between sentences using the cosine similarity function." ]
                , pre []
                    [ code [ class "python" ] python_sentence_sim_code ]
                , p []
                    [ text "The above example imports the tentence transformers library ("
                    , code [] [ text "pip install sentence-transfomers" ]
                    , text "). "
                    , text "We use the "
                    , code [] [ text "paraphrase-Mini-L6-v2" ]
                    , text " model for calculating the encodings for each sentence. After applying the cosine similarity function we get a score of 0.875 which indicates the two sentences are very similar."
                    ]
                , p [] [ text "Note that the similarity score will differ depending on the model we select; however, maximizing the accuracy is not the point of this blog. I am more interested in scaling the solution to large datasets." ]
                ]
            , section [ class "space-y-2" ]
                [ h2 [] [ text "Large Scale Solution" ]
                , p []
                    [ text "Suppose you have a list of documents (could be tens of thoushands) and you would like to compare all of them against each other. It would be very simple to get the vector encodings and do an element wise compairson. Below is the code; however, it is slow and redundant."
                    ]
                , pre []
                    [ code [ class "python" ]
                        [ text "def get_cosine_similarity(arr1, arr2):\n"
                        , text "\tnumerator = np.dot(arr1, arr2)\n"
                        , text "\tmag1 = np.sqrt(np.sum(np.square(arr1)))\n"
                        , text "\tmag2 = np.sqrt(np.sum(np.square(arr2)))\n"
                        , text "\treturn numerator / (mag1*mag2)"
                        ]
                    , code [ class "python" ]
                        [ text "def get_document_similarities(model, documents: List[str]):\n"
                        , text "\tsimilarities = [[0 for _ in range(len(documents))] \\"
                        , text "\n\t\t\t\t\tfor _ in range((len(documents)))]\n"
                        , text "\tfor i, doc1 in enumerate(documents):\n"
                        , text "\t\tdoc1_enc = model.encode(doc1)\n"
                        , text "\t\tfor j, doc2 in enumerate(documents):\n"
                        , text "\t\t\tdoc2_enc = model.encode(doc2)\n"
                        , text "\t\t\tsim = get_cosine_similarity(doc1_enc, doc2_enc)\n"
                        , text "\t\t\tsimilarities[i][j] = sim\n"
                        , text "\treturn similarities"
                        ]
                    ]
                , text "The complexity of the above algorithm is "
                , text "O(n^2)"
                , text " in terms of calculating the vector encodings for each document, assuming calculating the encodings and the cosine similarity is constant time complexity."
                , text "Although we cannot get past the "
                , text "O(n^2)"
                , text " limit set on our time complexity we can definitely make the algorithm faster."
                , ol
                    []
                    [ li [] [ text "The encodings for each document is being calculated more than once. We can pre-compute that." ]
                    , li []
                        [ text
                            "We are using python to calculate the cosine similarity. Using underlying numpy's matrix multiplication will also make the computation faster."
                        ]
                    ]
                ]
            , section []
                [ h3 [] [ text "Part 1: Get rid of Redundancy" ]
                , p [] [ text "Cosine simlarity is order independent, i.e.", code [] [ text "similarity(doc1, doc2) = similarity(doc2, doc1)" ], text "." ]
                , text "So if you have "
                , text "n"
                , text " documents, you do not want to compare against itself either. Thus the number computations reduce by "
                , text "\\frac{1}{2}"
                , text ". It becomes: "
                , text "(n^2-1) / 2"
                , pre []
                    [ code [ class "python" ]
                        [ text "def get_document_similarities(model, documents: List[str]):\n"
                        , text "\tsimilarities = [[0 for _ in range(len(documents))] \\"
                        , text "\n\t\t\t\t\tfor _ in range((len(documents)))]\n"
                        , text "\tfor i, doc1 in enumerate(documents):\n"
                        , text "\t\tdoc1_enc = model.encode(doc1)\n"
                        , text "\t\tfor j, doc2 in enumerate(documents):\n"
                        , text "\t\t\tif i == j:\n"
                        , text "\t\t\t\tsimilarities[i][j] = 1\n"
                        , text "\t\t\telif i>j:\n"
                        , text "\t\t\t\tdoc2_enc = model.encode(doc2)\n"
                        , text "\t\t\t\tsim = get_cosine_similarity(doc1_enc, doc2_enc)\n"
                        , text "\t\t\t\tsimilarities[i][j] = sim\n"
                        , text "\t\t\t\tsimilarities[j][i] = sim\n"
                        , text "\t\t\telif i<j:\n"
                        , text "\t\t\t\tcontinue\n"
                        , text "\treturn similarities"
                        ]
                    ]
                , p [] [ text "Problems," ]
                , ul []
                    [ li [] [ text "Redundant document encoding can be fixed with a hashmap" ]
                    , li []
                        [ text "Does not leverage fast C libraries for fast operations. Can be fixed with numpy arrays and matrix manipulation"
                        ]
                    ]
                ]
            , section []
                [ h3 [] [ text "Part 2: Matrix Multiplication" ]
                , p []
                    [ text "Let stack all document encodings rowise within a matrix. To make it simpler we will use a 2d vector representation for each document (in practice, as above, these dimension scale to 600+)."
                    , text "Let, "
                    , text "x = [a,b]"
                    , text " and "
                    , text "y = [c,d]"
                    , text "."
                    ]
                , text "We define out "
                , text "A"
                , text " matrix as a stacing our "
                , text "x"
                , text ", and "
                , text "y"
                , text " vectors."
                , case Array.get 1 from_arr of
                    Just val ->
                        div [] val

                    Nothing ->
                        div [] [ text "" ]
                , text "Lets multiply "
                , text "A"
                , text " by "
                , text "A^T"
                , case Array.get 2 from_arr of
                    Just val ->
                        div [] val

                    Nothing ->
                        div [] [ text "" ]
                , p []
                    [ text "Notice that this gives us the numerator portion of our cosine function. Why does this work? This is because the numerator is a dot product operation. That is what matrix multiplication is. A dot product!"
                    ]
                , text "All that is left now is to get the denominator and do an element wise division with our "
                , text "A"
                , text " matrix to get the cosine similarities. "
                , p []
                    [ text "In order to get the denominator for each pair of vectors, we first need to get the magnitude of each element. That works as follows. " ]
                , p []
                    [ text "Square each element"
                    ]
                , case Array.get 3 from_arr of
                    Just val ->
                        div [] val

                    Nothing ->
                        div [] [ text "" ]
                , text "Sum all rows in the "
                , text "A^2"
                , text " matrix and set the resulting vector to "
                , text "b"
                , text "."
                , case Array.get 4 from_arr of
                    Just val ->
                        div [] val

                    Nothing ->
                        div [] [ text "" ]
                , text "The computation of "
                , text "b \\cdot b^T"
                , text " is the resulting matrix for magnitudes of each pair of vectors. "
                , case Array.get 5 from_arr of
                    Just val ->
                        div [] val

                    Nothing ->
                        div [] [ text "" ]
                , text "Finally the resulting similarity matrix is the element wise multiplication of our "
                , text "A"
                , text " matrix and our reciprocal"
                , text "\\frac{1}{B}"
                , text " matrix."
                , case Array.get 6 from_arr of
                    Just val ->
                        div [] val

                    Nothing ->
                        div [] [ text "" ]
                , p [] [ text "Cosine similarity of all sentences. Loose redundancy and repetition; however, when performing matrix batch operations (which have been heavily optimized) is much faster. " ]
                , pre []
                    [ code [ class "python" ]
                        [ text """def cosine_similarity_faster(model, documents: List[str]):
\tdocument_encodings = np.array([model.encode(doc) for doc in documents])
\tnumerator = np.matmul(document_encodings, document_encodings.T)
\trow_sum = np.sqrt(np.sum(np.square(document_encodings), axis=1, keepdims=True))
\tdenominator = np.matmul(row_sum, row_sum.T)
\treturn numerator / denominator # will be done elementwise 
\t
cosine_similarity_faster(model, 
\t['The quick brown fox jumps over the lazy dogs',
\t'The quick fox jumped over the dogs']
)
'''
array([[1.0000004 , 0.87545466],
       [0.87545466, 1.0000004 ]], dtype=float32)
'''"""
                        ]
                    ]
                ]
            , section []
                [ h3 [ class "text-gray-600" ] [ text "Make it Faster Part 3" ]
                , p [] [ text "In an upcoming blog, we will look at making the above algorithm much faster, and use a real life dataset to show how powerful this implementation is." ]
                , p [ class "text-right" ]
                    [ a [ class "text-indigo-600", href "/cosine-similarity-pt2" ] [ text "To be Continued" ]
                    ]
                ]
            ]
        ]


python_sentence_sim_code : List (Html Msg)
python_sentence_sim_code =
    [ text "from sentence_transformers import SentenceTransformer\n"
    , text "# The Deep Learning model that will give us the vector representation\n"
    , text "model = SentenceTransformer('paraphrase-MiniLM-L6-v2')\n"
    , text "# The two sentences we want to compare\n"
    , text "s1 = 'The quick brown fox jumps over the lazy dogs'\n"
    , text "s2 = 'The quick fox jumped over the dogs'\n"
    , text "# Apply the model to get the number representation\n"
    , text "s1_encoding = model.encode(s1)\n"
    , text "s2_encoding = model.encode(s2)\n"
    , text "mag_s1 = np.sqrt(np.sum(np.square(s1_encoding)))\n"
    , text "mag_s2 = np.sqrt(np.sum(np.square(s2_encoding)))\n"
    , text "np.dot(s1_encoding, s2_encoding) / (mag_s1 * mag_s2)\n"
    , text "# Output: 0.8754566\n"
    ]
