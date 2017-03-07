# NodeJS Question Collections 

QUID : 0001
Modi : 2016-12-21 17:00:00
Chec : 2016-12-21 17:00:00
Tags : null
Prob : Cross origin requests are only supported for HTTP.
Desc : loading app.js in chrome would enconter such error. 
Solu : python2 -m SimpleHTTPServer at TARGET-folder to make app.js loadable.
Solu : python3 -m http.server 8008


QUID : 0002 
If brew install node not linked 
sudo brew link --overwrite node


// Merge-Tag Suggestion
// Tag-Mind-Map Suggestion
// Help you memory, help you learn, 

QUID : 0003
Webpack :: 
npm install webpack -g :: able to use in CLI
npm install webpack --save-dev :: As building process dependency
npm install webpack --save :: As must dependency


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


