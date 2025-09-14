// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-publications",
          title: "publications",
          description: "publications in reversed chronological order.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-repositories",
          title: "repositories",
          description: "my repos",
          section: "Navigation",
          handler: () => {
            window.location.href = "/repositories/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: ".",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "dropdown-bookshelf",
              title: "bookshelf",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/books/";
              },
            },{id: "dropdown-blog",
              title: "blog",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/blog/";
              },
            },{id: "post-debugging-numeric-comparisons-llms",
        
          title: "Debugging Numeric Comparisons Llms",
        
        description: "Layerwise geometry shows the model internally separates Yes/No, but a last-layer readout corrupts the decision—especially for decimals.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/debugging-numeric-comparisons-llms/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "news-ranked-7th-in-jane-street-market-prediction-kaggle-challenge-among-4245-teams",
          title: 'Ranked 7th in Jane Street Market Prediction kaggle challenge among 4245 teams.',
          description: "",
          section: "News",},{id: "news-joined-millennium-management-as-quantitative-researcher",
          title: 'Joined Millennium Management as Quantitative Researcher.',
          description: "",
          section: "News",},{id: "news-short-paper-accepted-in-ieee-hpcc-2021",
          title: 'Short Paper Accepted in IEEE HPCC-2021',
          description: "",
          section: "News",},{id: "news-our-paper-using-shapley-interactions-to-understand-how-models-use-structure-got-published-in-acl-2025-main",
          title: 'Our paper Using Shapley interactions to understand how models use structure got published...',
          description: "",
          section: "News",},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%64%69%76%79%61%6E%73%68%73%69%6E%67%68%76%69@%67%6D%61%69%6C.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/feed.xml", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=7AaPJF8AAAAJ", "_blank");
        },
      },{
        id: 'social-custom_social',
        title: 'Custom_social',
        section: 'Socials',
        handler: () => {
          window.open("https://www.divyanshsinghvi.github.io/", "_blank");
        },
      },];
