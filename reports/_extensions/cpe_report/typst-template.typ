
// This is an example typst template (based on the default template that ships
// with Quarto). It defines a typst function named 'article' which provides
// various customization options. This function is called from the
// 'typst-show.typ' file (which maps Pandoc metadata function arguments)
//
// If you are creating or packaging a custom typst template you will likely
// want to replace this file and 'typst-show.typ' entirely. You can find
// documentation on creating typst templates and some examples here:
//   - https://typst.app/docs/tutorial/making-a-template/
//   - https://github.com/typst/templates


#let article(
  title: none,
  subtitle: none,
  authors: none,
  date: none,
  abstract: none,
  abstract-title: none,
  cols: 1,
  margin: (x: 1.25in, y: 1.25in),
  paper: "us-letter",
  lang: "en",
  region: "US",
  font: "libertinus serif",
  fontsize: 11pt,
  title-size: 1.5em,
  subtitle-size: 1.25em,
  heading-family: "libertinus serif",
  heading-weight: "bold",
  heading-style: "normal",
  heading-color: black,
  heading-line-height: 0.65em,
  sectionnumbering: none,
  pagenumbering: "1",
  toc: false,
  toc_title: none,
  toc_depth: none,
  toc_indent: 1.5em,
  doc,
) = {
  set page(
    paper: paper,
    margin: margin,
    numbering: none,
  )
  set par(justify: true)
  set text(
    lang: lang,
    region: region,
    font: font,
    size: fontsize,
  )
  set heading(numbering: sectionnumbering)

  let title_text = [
    #if title != none {
      text(weight: "bold", size: title-size)[#title]
      if subtitle != none {
        parbreak()
        text(weight: "bold", size: subtitle-size)[#subtitle]
      }
    }

    #v(1fr)

    // Author names
    #if authors != none {
      for author in authors [
        #author.name
        #linebreak()
      ]
    }

    #v(1fr)

    Department of Computer Engineering \
    Faculty of Engineering \
    King Mongkut's University Of Technology Thonburi
  ]

  align(center)[
    #image("./figures/kmutt_logo.jpg", width: 2.8cm)
    #v(1em)
    #title_text
    #v(2cm)
  ]

  pagebreak()
  set page(
    paper: paper,
    margin: margin,
    numbering: "(i)",
  )

  if abstract != none {
    block(inset: 2em)[
      #text(weight: "semibold")[#abstract-title] #h(1em) #abstract
    ]
  }

  if toc {
    let title = if toc_title == none {
      auto
    } else {
      toc_title
    }
    block(above: 0em, below: 2em)[
      #outline(
        title: toc_title,
        depth: toc_depth,
        indent: toc_indent,
      );
    ]
  }

  set page(
    paper: paper,
    margin: margin,
    numbering: pagenumbering,
  )
  counter(page).update(1)

  if cols == 1 {
    doc
  } else {
    columns(cols, doc)
  }
}

#set table(
  inset: 6pt,
  stroke: none,
)
