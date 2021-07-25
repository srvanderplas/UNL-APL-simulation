#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#
library(tidyverse)
library(shiny)

#--- UI definition -------------------------------------------------------------
# Define UI for application that draws a histogram
ui <- navbarPage(
    windowTitle = "APL Statistics",
    title = "APL Simulations",  
    tabPanel(
        icon = icon("coffee"), 
        title = "Tea Tasting",
        fluidRow(
            column( 
                width = 12,
                wellPanel(
                    h3("A Lady Tasting Tea"),
                    "Problem description, link to wikipedia, etc."
                )
            )
        ),
        fluidRow(
            column(
                width = 3, 
                wellPanel(
                    h3("Set up"),
                    numericInput("ncups", "Number of cups of each type", value = 4, min = 2, max = 10),
                    numericInput("ntrials", "Number of simulations", value = 100, min = 25, max = 1000),
                    numericInput("prob", "Probability of guessing correctly", value = .5, min = 0, max = 1, step = .1),
                    helpText("Under random chance, this should be 0.5"),
                    numericInput("obs", "Number of correct guesses of each type (Observed)", value = 4, min = 0, max = 10)
                )
            ),
            column(
                width = 9, 
                h3("Experiment Results"),
                plotOutput("tea_plot"),
                uiOutput("tea_res")
            )
        )
    ),
    tabPanel(
        icon = icon("wrench"),
        title = "Distributions",
        h2("Sampling vs. Theory-based Statistics"),
        fluidRow(
            column( 
                width = 12,
                wellPanel(
                    h3("Setting up the problem"),
                    "Problem description, link to wikipedia, etc."
                )
            )
        ),
        fluidRow(
            column(
                width = 3, 
                wellPanel(
                    h3("Choose your distribution and parameters"),
                    selectInput(
                        inputId = "dname", label = "Distribution",
                        choices = c("Normal" = "z", "T" = "t", "Chi-squared" = "chisq", "F" = "f"),
                        selected = 1
                    ),
                    numericInput("dist.n", label = "Number of samples", 
                                 value = 100, min = 30, max = 1000),
                    numericInput("dist.observed", label = "Observed value", value = 2),
                    conditionalPanel(
                        "input.dname == 'z'",
                        numericInput("dist.mean", label = "Mean", value = 0),
                        numericInput("dist.sd", label = "Std Dev", value = 1, min = 0.01)
                    ),
                    conditionalPanel(
                        "input.dname == 'chisq' | input.dname == 't'",
                        numericInput("dist.df", label = "Degrees of Freedom (DF)", value = 10, step = 1, min = 1)
                    ),
                    conditionalPanel(
                        "input.dname == 'f'",
                        numericInput("dist.df1", label = "Numerator df", value = 10, step = 1, min = 1),
                        numericInput("dist.df2", label = "Denominator df", value = 2, step = 1, min = 1)
                    )
                )
            ),
            column(
                width = 9, height = 500,
                fluidRow(
                    column(
                        width = 6,
                        h3("Simulation Results"),
                        # dataTableOutput("dists_sim_data"),
                        plotOutput("dists_sim_plot"),
                        uiOutput("dists_sim_text")
                    ),
                    column(
                        width = 6,
                        h3("Theoretical Results"),
                        plotOutput("dists_theory_plot"),
                        textOutput("dists_theory_text")
                    )
                )
            )
        )
    ),
    tabPanel(
        title = "One Categorical Variable",
        h2("Studies with One Categorical Variable"),
        fluidRow(
            column( 
                width = 12,
                wellPanel(
                    h3("Setting up the problem"),
                    "Problem description, link to wikipedia, etc."
                )
            )
        ),
        fluidRow(
            column(
                width = 3, 
                wellPanel(
                    h3("Set your observed values and simulation parameters"),
                    numericInput("oc.observed", label = "Observed value", value = 16),
                    numericInput("oc.total", label = "Total # Trials", value = 20),
                    br(),
                    numericInput("oc.numberSims", label = "# Simulations to run", 
                                 value = 100, min = 10, max = 1000),
                    numericInput("oc.simprob", label = "Simulation success probability",
                                 value = .5, min = 0, max = 1)
                )
            ),
            column(
                width = 9, height = 500,
                fluidRow(
                    column(
                        width = 6,
                        h3("Simulation Results"),
                        plotOutput("oc_sim_plot"),
                        uiOutput("oc_sim_text")
                    ),
                    column(
                        width = 6,
                        h3("Theoretical Results"),
                        plotOutput("oc_theory_plot"),
                        textOutput("oc_theory_text")
                    )
                )
            )
        )
    ),
    tabPanel(
        title = "One Continuous Variable",
        h2("Studies with One Continuous Variable"),
        fluidRow(
            column( 
                width = 12,
                wellPanel(
                    h3("Setting up the problem"),
                    "Problem description, link to wikipedia, etc."
                )
            )
        ),
        fluidRow(
            column(
                width = 3, 
                wellPanel(
                    h3("Set your observed values and Null hypothesis"),
                    numericInput("octs.samplemean", label = "Sample Mean", value = 10.5),
                    numericInput("octs.samplesd", label = "Sample SD", value = 2),
                    numericInput("octs.n", label = "Sample size", value = 50, min = 10),
                    br(),
                    numericInput("octs.distmean", label = "Hypothesized mean (mu)", 
                                 value = 10),
                    selectInput("octs.hypothesis", label = "Null Hypothesis Type",
                                choices = c("x > mu", "x < mu", "x = mu"))
                )
            ),
            column(
                width = 9, 
                h3("Theoretical Results"),
                helpText("Note that with continuous variables, we cannot resample data without knowledge of the underlying distribution or much larger datasets (e.g. a census)."),
                plotOutput("octs_theory_plot"),
                textOutput("octs_theory_text")
            )
        )
    ),
    tabPanel(
        title = "Categorical + Continuous Variables",
        h2("Studies with Categorical + Continuous Variables"),
        fluidRow(
            column( 
                width = 12,
                wellPanel(
                    h3("Setting up the problem"),
                    "Problem description, link to wikipedia, etc."
                )
            )
        ),
        fluidRow(
            column(
                width = 3, 
                wellPanel(
                    h3("Set your observed values and simulation parameters"),
                    fluidRow(
                        column(
                            width = 6, 
                            textAreaInput("tg.group1", label = "Group 1 Data", 
                                          value = paste(round(rnorm(20, 5, 1), 3), 
                                                        collapse = "\n"),
                                          height = "200px"),
                            uiOutput("group1stats")
                        ),
                        column(
                            width = 6, 
                            textAreaInput("tg.group2", label = "Group 2 Data", 
                                          value = paste(round(rnorm(25, 7, 1.5), 3), 
                                                        collapse = "\n"),
                                          height = "200px"),
                            uiOutput("group2stats")
                        )
                    ),
                    br(),
                    withMathJax(
                        helpText("The statistic calculated will be \\(\\overline x_1 - \\overline x_2\\)"),
                        helpText("We will use a null hypothesis of \\(\\mu_1 - \\mu_2 = 0\\)")),
                    numericInput("tg.numberSims", label = "# Simulations to run", 
                                 value = 100, min = 10, max = 1000)
                )
            ),
            column(
                width = 9,
                fluidRow(
                    column(
                        width = 6,
                        h3("Simulation Results"),
                        plotOutput("tg_sim_plot"),
                        uiOutput("tg_sim_text")
                    ),
                    column(
                        width = 6,
                        h3("Theoretical Results"),
                        plotOutput("tg_theory_plot"),
                        verbatimTextOutput("tg_theory_text")
                    )
                )
            )
        )
    )
)
#--- Server definition ---------------------------------------------------------

# Define server logic required to draw a histogram
server <- function(input, output) {
    tea_sim <- reactive({
        validate(need(input$obs <= input$ncups, "Correct guesses must be less than or equal to number of cups"))
        res <- tibble(x = rbinom(input$ntrials, input$ncups, input$prob))
        res$fill <- factor(ifelse(res$x >= input$obs, "As extreme", "Not as extreme"))
        
        res
    })
    output$tea_plot <- renderPlot({
        ggplot(data = tea_sim(), aes(x = x, fill = fill)) + geom_bar(color = "black") + 
            scale_fill_manual("Compared to\nObserved Value", 
                              values = c("Not as extreme" = "white", "As extreme" = "red")) +
            xlab("# Correct Cups Identified") + ylab("# Simulations") + 
            ggtitle("Results of Tea Test Under Random Chance") + 
            theme(legend.position = c(1, 1), legend.justification = c(1,1), legend.background = element_rect(fill = "transparent", color = "black"))
    })
    
    output$tea_res <- renderUI({
        HTML(
            paste(
                sprintf("Number of simulations with at least %d correct guesses: %d", input$obs, sum(tea_sim()$fill == "As extreme")),
                sprintf("P-value: %0.3f", mean(tea_sim()$fill == "As extreme")),
                sep = "<br/>"
            )
        )
    })
    
    dists_res <- reactive({
        tibble(dist = c("z", "t", "chisq", "f"),
               name = c("Normal", "T", "Chi-squared", "F"),
               rdmfun = c(rnorm, rt, rchisq, rf),
               densfun = c(dnorm, dt, dchisq, df),
               pfun = c(pnorm, pt, pchisq, pf)) %>%
            filter(dist == input$dname)
    })
    
    dist_args <- reactive({
        args <- list(
            n = ifelse("dist.n" %in% names(input), input$dist.n, NULL),
            mean = ifelse("dist.mean" %in% names(input), input$dist.mean, NULL),
            sd = ifelse("dist.sd" %in% names(input), input$dist.sd, NULL),
            df = ifelse("dist.df" %in% names(input), input$dist.df, NULL),
            df1 = ifelse("dist.df1" %in% names(input), input$dist.df1, NULL),
            df2 = ifelse("dist.mean" %in% names(input), input$dist.df2, NULL)
        )
        # args <- args[-which(sapply(args, is.null))] 
        
        if (input$dname == 'z') {
            args <- args[names(args) %in% c("n", "mean", "sd")]
        } else if (input$dname == 't') {
            args <- args[names(args) %in% c("n", "df")]
        } else if (input$dname == 'chisq') {
            args <- args[names(args) %in% c("n", "df")]
        } else if (input$dname == 'f') {
            args <- args[names(args) %in% c("n", "df1", "df2")]
        }
        
        args
    })
    dists_sim_res <- reactive({
        data <- do.call(dists_res()$rdmfun[[1]], dist_args())
        tibble(x = data, extreme = factor(ifelse(data >= input$dist.observed, "As extreme", "Not as extreme")))
    })
    
    output$dists_sim_data <- renderDataTable({
        dists_sim_res()
    })
    
    output$dists_sim_plot <- renderPlot({
        validate(need(nrow(dists_sim_res()) > 0, "Data simulation did not succeed"))
        validate(need(nrow(dists_res()) > 0, "Function identification did not succeed"))
        
        ggplot(data = dists_sim_res(), aes(x = x, fill = extreme)) + 
            geom_histogram(color = "black") + 
            scale_fill_manual("Compared to\nObserved Value", 
                              values = c("Not as extreme" = "white", "As extreme" = "red")) +
            ylab("# Simulations") +
            ggtitle(sprintf("Simulated %s Values", dists_res()$name[[1]])) +
            theme(legend.position = c(1, 1), legend.justification = c(1,1), 
                  legend.background = element_rect(fill = "transparent", 
                                                   color = "black"), 
                  axis.title.x = element_blank())
    })
    
    output$dists_sim_text <- renderUI({
        paste(
            sprintf("Number of simulations with values as or more extreme than %0.2f: %d", 
                        input$dist.observed, sum(dists_sim_res()$extreme == "As extreme")),
            sprintf("Simulation P-value: %0.3f", mean(dists_sim_res()$extreme == "As extreme")),
            sep = "<br/><br/>"
        ) %>% HTML()
    })
    
    output$dists_theory_plot <- renderPlot({
        args <- c(dist_args())
        args <- args[-which(names(args) == "n")]
        domain.lim <- max(c(1.5*abs(input$dist.mean), 
                           1.5*abs(input$dist.observed), 
                           abs(dists_sim_res()$x)))
        domain <- if ( input$dname %in% c('z', 't')) {
            c(-domain.lim, domain.lim)
        } else {
            c(0, domain.lim)
        }
        ggplot() + 
            xlim(domain) + 
            stat_function(fun = dists_res()$densfun[[1]], 
                          args = args, geom = "area", 
                          color = "black", fill = "white"
                          ) + 
            stat_function(fun = dists_res()$densfun[[1]], args = args, 
                          xlim = c(input$dist.observed, max(domain)), geom = "area",
                          color = "black", fill = "red"
                          ) + 
            scale_fill_discrete("Compared to\nObserved Value") +
            ylab("Density") +
            ggtitle(sprintf("Theoretical %s Distribution", dists_res()$name[[1]])) +
            theme(legend.position = c(1, 1), legend.justification = c(1,1), 
                  legend.background = element_rect(fill = "transparent", 
                                                   color = "black"))
    })
    output$dists_theory_text <- renderText({
        
        args <- c(q = input$dist.observed, dist_args(), lower.tail = F)
        args <- args[-which(names(args) == "n")]
        sprintf(
            "Theoretical P-value: %0.3f", 
            do.call(dists_res()$pfun[[1]], args))
    })
    
    oc_sim <- reactive({
        tibble(
            x = rbinom(input$oc.numberSims, size = input$oc.total, prob = input$oc.simprob),
            extreme = factor(ifelse(x >= input$oc.observed, "As extreme", "Not as extreme")))
    })
    
    output$oc_sim_plot <- renderPlot({
        # print(oc_sim())
        ggplot(data = oc_sim(), aes(x = x, fill = extreme)) + 
            geom_bar(color = "black") + 
            scale_fill_manual("Compared to\nObserved Value", 
                              values = c("Not as extreme" = "white", "As extreme" = "red")) +
            ylab("# Simulations") +
            ggtitle("Simulated Successes") +
            theme(legend.position = c(1, 1), legend.justification = c(1,1), 
                  legend.background = element_rect(fill = "transparent", 
                                                   color = "black"), 
                  axis.title.x = element_blank())
        
    })    
    output$oc_sim_text <- renderUI({
        paste(
            sprintf("Number of simulations with at least %d successes: %d", 
                    input$oc.observed, sum(oc_sim()$extreme == "As extreme")),
            sprintf("Simulation P-value: %0.3f", mean(oc_sim()$extreme == "As extreme")),
            sep = "<br/><br/>"
        ) %>% HTML()
    })
    output$oc_theory_plot <- renderPlot({
        data <- tibble(
            x = seq(0, input$oc.total, 1),
            y = dbinom(x = x, size = input$oc.total, prob = input$oc.simprob),
            extreme = factor(ifelse(x >= input$oc.observed, "As extreme", "Not as extreme"))
        )
        
        
        ggplot(data) + 
            geom_bar(aes(x = x, y = y, fill = extreme), stat = "identity", color = "black") +
            scale_fill_manual("Compared to\nObserved Value", 
                              values = c("Not as extreme" = "white", "As extreme" = "red")) +
            ylab("Probability") +
            ggtitle("Theoretical Distribution") +
            theme(legend.position = c(1, 1), legend.justification = c(1,1), 
                  legend.background = element_rect(fill = "transparent", 
                                                   color = "black"))
        
    })
    
    output$oc_theory_text <- renderText({
        sprintf(
            "Theoretical P-value: %0.3f", 
            pbinom(input$oc.observed, input$oc.total, input$oc.simprob, lower.tail = F))
    })
    
    
    output$octs_theory_plot <- renderPlot({
        if (input$octs.hypothesis == "x > mu") {
            t_stat <- (input$octs.samplemean - input$octs.distmean)/(input$octs.samplesd/sqrt(input$octs.n))
            xlimshade <- c(t_stat, 4)
            side2 <- F
        } else if (input$octs.hypothesis == "x < mu") {
            t_stat <- -(input$octs.samplemean - input$octs.distmean)/(input$octs.samplesd/sqrt(input$octs.n))
            xlimshade <- c(-4, t_stat)
            side2 <- F
        } else {
            t_stat <- (input$octs.samplemean - input$octs.distmean)/(input$octs.samplesd/sqrt(input$octs.n))
            xlimshade <- c(t_stat, 4)
            side2 <- T
        }
        degf <- input$octs.n - 1
        
        plot <- ggplot() + 
            # xlim(c(-4,4)) + 
            stat_function(fun = dt, 
                          args = c(df = degf), geom = "area", 
                          color = "black", fill = "white",
                          xlim = c(-4, 4)
            ) + 
            stat_function(fun = dt, args = c(df = degf), 
                          xlim = xlimshade, geom = "area",
                          color = "black", fill = "red"
            ) + 
            geom_vline(aes(xintercept = t_stat)) + 
            geom_text(aes(x = t_stat + .05, y = Inf, 
                          label = sprintf("Calculated t-value\nt = %0.2f", t_stat)), 
                      vjust = 2, hjust = 0) + 
            ylab("Density") +
            ggtitle("Theoretical T-Distribution") +
            theme(legend.position = c(1, 1), legend.justification = c(1,1), 
                  legend.background = element_rect(fill = "transparent", 
                                                   color = "black"),
                  axis.title.x = element_blank())
        
        if (side2) {
            plot <- plot +
                stat_function(fun = dt, args = c(df = degf), 
                              xlim = -rev(xlimshade), geom = "area",
                              color = "black", fill = "red"
                ) + 
                geom_vline(aes(xintercept = -t_stat)) 
        }
        plot
        
    })

    output$octs_theory_text <- renderText({
        t_stat <- (input$octs.samplemean - input$octs.distmean)/(input$octs.samplesd/sqrt(input$octs.n))
        degf <- input$octs.n - 1
        if (input$octs.hypothesis == "x = mu") {
            pv <- pt(t_stat, df = degf, lower.tail = F) * 2
        } else {
            pv <- pt(t_stat, df = degf, lower.tail = F)
        }
        sprintf("Theoretical P-value: %0.3f", pv)
    })
    
    tg_data <- reactive({
        g1 <- str_split(input$tg.group1, "\\s", simplify = T) %>% as.numeric()
        g2 <- str_split(input$tg.group2, "\\s", simplify = T) %>% as.numeric()
        tibble(
            group = c(rep(1, length(g1)), rep(2, length(g2))),
            x = c(g1, g2)
            )
        })
    
    shuffle_groups <- function(data) {
        data %>%
            mutate(group = sample(group, length(group), replace = F))
    }
    
    summarize_groups <- function(data) {
        data %>%
            group_by(group) %>%
            summarize(x = mean(x)) %>%
            arrange(group) %>%
            select(x) %>%
            unlist() %>%
            as.numeric() %>%
            diff()
    }
    tg_sim <- reactive({
        actual_diff <- tg_data() %>%
            summarize_groups()
        
        tibble(
            diff = purrr::map_dbl(1:input$tg.numberSims, ~shuffle_groups(tg_data()) %>% summarize_groups()),
            extreme = abs(diff) >= abs(actual_diff)
        ) %>%
            mutate(extreme = factor(extreme, levels = c(FALSE, TRUE), labels = c("Not as extreme", "As extreme")))
    })
    
    output$group1stats <- renderUI({
        data <- filter(tg_data(), group == 1)$x
        sprintf("Group 1 Mean: %0.2f<br/>Group 1 SD: %0.2f<br/>Group 1 N: %d",
                mean(data), sd(data), length(data)) %>% HTML()
    })
    
    output$group2stats <- renderUI({
        data <- filter(tg_data(), group == 2)$x
        sprintf("Group 2 Mean: %0.2f<br/>Group 2 SD: %0.2f<br/>Group 2 N: %d",
                mean(data), sd(data), length(data)) %>% HTML()
    })
    
    # tg_sim <- reactive()
    output$tg_sim_plot <- renderPlot({
        obs_stat <- tg_data() %>%
            summarize_groups()
        diff_lim <- max(c(obs_stat, tg_sim()$x))
        
        ggplot(data = tg_sim(), aes(x = diff, fill = extreme)) + 
            xlim(1.2*c(-diff_lim, diff_lim)) +
            geom_histogram(color = "black") + 
            geom_vline(aes(xintercept = obs_stat)) + 
            geom_vline(aes(xintercept = -obs_stat)) + 
            scale_fill_manual("Compared to\nObserved Value",
                              values = c("Not as extreme" = "white", "As extreme" = "red")) +
            ylab("# Simulations") +
            ggtitle("Simulated Successes") +
            theme(legend.position = c(1, 1), legend.justification = c(1,1), 
                  legend.background = element_rect(fill = "transparent", 
                                                   color = "black"), 
                  axis.title.x = element_blank())
        
    })    
    output$tg_sim_text <- renderUI({
        obs_stat <- tg_data() %>%
            summarize_groups()
        
        paste(
            sprintf("Number of simulations with a difference of at least +/- %0.3f", 
                    obs_stat),
            sprintf("Simulation P-value: %0.3f", mean(tg_sim()$extreme == "As extreme")),
            sep = "<br/><br/>"
        ) %>% HTML()
    })
    
    output$tg_theory_plot <- renderPlot({
        ttest <- t.test(x ~ group, data = tg_data())
        
        lims <- max(c(abs(ttest$statistic) + 1, 4))
        
        ggplot() + 
            xlim(c(-lims, lims)) +
            stat_function(fun = dt, 
                          args = c(df = ttest$parameter[[1]]), geom = "area", 
                          color = "black", fill = "white"
            ) + 
            stat_function(fun = dt, args = c(df = ttest$parameter[[1]]), 
                          xlim = c(-lims, -abs(ttest$statistic)), geom = "area",
                          color = "black", fill = "red"
            ) + 
            stat_function(fun = dt, args = c(df = ttest$parameter[[1]]), 
                          xlim = c(abs(ttest$statistic), lims), geom = "area",
                          color = "black", fill = "red"
            ) + 
            geom_vline(aes(xintercept = ttest$statistic)) + 
            geom_vline(aes(xintercept = -ttest$statistic)) + 
            ylab("Density") +
            ggtitle("Theoretical T-Distribution") +
            theme(legend.position = c(1, 1), legend.justification = c(1,1), 
                  legend.background = element_rect(fill = "transparent", 
                                                   color = "black"),
                  axis.title.x = element_blank())
        
    })
    
    output$tg_theory_text <- renderPrint({
        ttest <- t.test(x ~ group, data = tg_data())
        
        print(ttest)
    })
    
}


# Run the application 
shinyApp(ui = ui, server = server)

