module Helper exposing (..)

import Html exposing (..)
import Html.Attributes exposing (class)



{- import Katex as K
   exposing
       ( Latex
       , display
       , human
       , inline
       )
-}


type BlogDevelopmentStep
    = Draft String
    | BlogReady


type alias Date =
    { month : String, date : String, year : Int }


type PulicationDate
    = InDevelopment
    | Publised Date


type alias BlogPostMetaData =
    { title : String
    , published_date : PulicationDate -- String
    , post_link : String
    , summary : String
    , developmentStep : BlogDevelopmentStep
    }


blog_section : List (Html msg) -> Html msg
blog_section children =
    section [] children


blog_p : List (Html msg) -> Html msg
blog_p children =
    p [] children


python_code_block : List (Html msg) -> Html msg
python_code_block code_list =
    pre [] [ code [ class "python" ] code_list ]


display_publication_data : PulicationDate -> String
display_publication_data pub_date =
    case pub_date of
        InDevelopment ->
            "TBD"

        Publised publised_date ->
            publised_date.month ++ " " ++ publised_date.date ++ ", " ++ String.fromInt publised_date.year



{-
   page_view_template : BlogPostMetaData -> Html msg -> Html msg
   page_view_template meta_data children =
       div
           [ class "mb-40" ]
           [ p [ class "text-gray-500 text-sm" ] [ text <| display_publication_data meta_data.published_date ]
           , p
               [ class "text-grey-600" ]
               [ span [ class "font-medium" ] [ text "Summary" ]
               , text ": "
               , text meta_data.summary
               , case meta_data.developmentStep of
                   Draft draft_v ->
                       div [ class "mt-3" ]
                           [ span [ class "text-indigo-900 bg-indigo-200 px-2 py-1 rounded-lg" ] [ text "draft", text " ", text draft_v ]
                           ]

                   _ ->
                       text ""
               ]
           , children
           ]

-}
{-
   compile_latex_code : List Latex -> Html a
   compile_latex_code lst =
       lst
           |> List.map (K.generate htmlGenerator)
           |> div [ class "" ]
-}


htmlGenerator : Maybe Bool -> String -> Html msg
htmlGenerator isDisplayMode stringLatex =
    case isDisplayMode of
        Just True ->
            div [ class "overflow-x-auto py-5" ] [ text stringLatex ]

        _ ->
            span [] [ text stringLatex ]
