const Algebra = await fetch("https://enkimute.github.io/ganja.js/ganja.js")
                      .then(x=>x.text())
                      .then(x=>{ const ctx = {}; (new Function(x)).apply(ctx); return ctx.Algebra });


function render({ model, el }) {
    var canvas = Algebra({metric: model.get('signature'), Cayley: model.get('cayley')}).inline((model)=>{
        var getSubjects = ()=>model.get('subjects');
        var options = model.get('options');
        var decode = x=>typeof x === 'object' && 'mv' in x?new Element(x['mv']):Array.isArray(x)?x.map(decode):x;

        var canvas = this.graph(()=>{
                model.send({ type: "update_mvs" });
                var subjects = getSubjects();
                console.log(subjects);
                return [...decode(subjects)];
            },
            options
        )
        return canvas;
    })(model)
    canvas.style.width = '100%';
    canvas.style.background = 'white';
    el.appendChild(canvas);
}

export default { render };
