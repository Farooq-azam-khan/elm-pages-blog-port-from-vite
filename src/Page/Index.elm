module Page.Index exposing (Data, Model, Msg, page)

import DataSource exposing (DataSource)
import Head
import Head.Seo as Seo
import Helper
    exposing
        ( BlogDevelopmentStep(..)
        , BlogPostMetaData
        , PulicationDate(..)
        , display_publication_data
        )
import Html exposing (..)
import Html.Attributes exposing (..)
import Page exposing (Page, StaticPayload)
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


data : DataSource Data
data =
    DataSource.succeed ()


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
        , description = "A Blog by me, Farooq Azam Khan."
        , locale = Nothing
        , title = "Farooq Azam Khan | Blog"
        }
        |> Seo.website


type alias Data =
    ()


view :
    Maybe PageUrl
    -> Shared.Model
    -> StaticPayload Data RouteParams
    -> View Msg
view maybeUrl sharedModel static =
    { title = "Farooq Azam Khan | Blog"
    , body =
        [ div
            [ class "mx-5 sm:mx-0 sm:mx-auto prose lg:prose-lg sm:max-w-xl lg:max-w-3xl mt-10 " ]
            [ div [ class "sm:flex sm:items-center space-x-3" ]
                [ a [ target "blank", class "flex-shrink-0", href "http://www.github.com/farooq-azam-khan" ]
                    [ img [ alt "image of Farooq Azam Khan", class "rounded-full w-32 h-32", src "https://avatars.githubusercontent.com/u/33574913?v=4" ]
                        []
                    ]
                , h1
                    [ class "hover:underline tracking-wide" ]
                    [ text "Welcome to my Blog" ]
                ]
            , home_page_content
            ]
        ]
    }


home_page_content : Html Msg
home_page_content =
    div [ class "" ]
        [ section [ class "space-y-2" ] (List.map blog_list_post_component blog_posts_lists)
        , section []
            [ h2 [ class "text-gray-700" ] [ text "Drafts" ]
            , ul [ class "list-disc" ]
                [ li [] [ text "Singular Value Decomposition and Recommendation Engines" ]
                , li [] [ text "What Principal Component Analysis teaches us about Dimensionality reduction" ]
                , li [] [ text "The Deep Learning Model Development Architecture" ]
                , li [] [ text "Linear Regression: The Basis for all Modern Deep Learning Algorithms" ]
                , li [] [ text "What RNNs are and why they are Turing Complete!" ]
                ]
            ]
        ]


blog_posts_lists : List BlogPostMetaData
blog_posts_lists =
    [ { title = "Term Frequency-Inverse Document Frequency"
      , published_date = InDevelopment
      , summary = "In this tutorial we will look at what TF and IDF are and how they can be use to process text data in Machine learning."
      , post_link = "tfidf"
      , developmentStep = Draft "1.0"
      }
    , { title = "Large Scale Vector Comparison"
      , published_date = Publised { month = "July", date = "9th", year = 2022 }
      , summary = "In this post, we will look at the quora qna dataset and aim to encode and compare all question pairs. The purpose of is to look at a real dataset."
      , post_link = "cosine-similarity-pt2"
      , developmentStep = BlogReady
      }
    , { title = "Comparing Vectors with Cosine Simlarity Function"
      , published_date = Publised { month = "July", date = "4th", year = 2022 }
      , summary = "This tutorial will focus on the math behind text vector similarity using numpy, pytorch, and stentence-transformers libraries in python."
      , post_link = "cosine-similarity"
      , developmentStep = BlogReady
      }
    ]


blog_list_post_component : BlogPostMetaData -> Html msg
blog_list_post_component blog_data =
    div
        [ class "hover:bg-orange-100 py-2 rounded hover:rounded-lg ease-in duration-200 border-l-4  border-white hover:border-indigo-400 px-3 flex flex-col space-y-2" ]
        [ span [ class "text-indigo-600 " ] [ text <| display_publication_data blog_data.published_date ]
        , span [ class "mt-3" ]
            [ a
                [ href blog_data.post_link ]
                [ text blog_data.title ]
            ]
        , span [ class "text-gray-700" ]
            [ text blog_data.summary
            ]
        ]
