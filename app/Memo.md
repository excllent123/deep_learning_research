# NodeJS Question Collections 

QUID : 0001
Modi : 2016-12-21 17:00:00
Chec : 2016-12-21 17:00:00
Tags : null
Prob : Cross origin requests are only supported for HTTP.
Desc : loading app.js in chrome would enconter such error. 
Solu : python -m SimpleHTTPServer at TARGET-folder to make app.js loadable.

QUID : 0002 

If brew install node not linked 
sudo brew link --overwrite node


// Merge-Tag Suggestion
// Tag-Mind-Map Suggestion
// Help you memory, help you learn, 

QUID : 
Webpack :: npm install webpack -g :: able to use in CLI

npm install webpack --save-dev :: As dependency


Prob : material-ui webpack syntax fail
Solu : babel-preset-stage-2 @ webpack
 - Prob : Unexpected extra options ["stage-2"] passed to preset.
 npm install --global babel-cli

Prob : this.context.muiTheme is undefined
Tags : material-ui, 
import getMuiTheme from 'material-ui/styles/getMuiTheme'
...


    static childContextTypes =
    {
        muiTheme: React.PropTypes.object
    }

    getChildContext()
    {
        return {
            muiTheme: getMuiTheme()
        }
    }


