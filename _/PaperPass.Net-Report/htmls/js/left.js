(function(M,D){D(function(){var o=typeof isHasDetailReport;var n;if(o=="boolean"&&isHasDetailReport==true){var e=typeof isWordDetailReport;if(e=="boolean"&&isWordDetailReport==true){var t=M.operateState.getDetailNum(Report.report_id);var i=M.operateState.getDetailOffset(Report.report_id);var a=M.operateState.getDetailSize(Report.report_id);var s=M.operateState.getZoomBtnState(Report.report_id);var r=M.operateState.getLineState(Report.report_id);n=true}else{var t=M.operateState.getPlainTextDetailNum(Report.report_id);var i=M.operateState.getPlainTextDetailOffset(Report.report_id);var a=M.operateState.getPlainTextDetailSize(Report.report_id);var s=M.operateState.getPlainTextZoomBtnState(Report.report_id);var r=M.operateState.getPlainTextLineState(Report.report_id);n=false}}var l=M.operateState.getLanguage(Report.report_id);if(l!=""){l=typeof multilingual[l]=="undefined"?"English":l}else{l=typeof multilingual[user_language]=="undefined"?"English":user_language}c(l);function c(t){D("*[data-lang]").each(function(){var e=D(this).data().lang;D(this).html(multilingual[t][e])})}parent.postMessage("get_language","*");parent.parent.postMessage("get_language","*");var u=`<span class="pageNum">1</span><span class="partition">/</span><span class="totalPage"></span>`;D(".top_Num").html(u);var p=D(".upkmzjakmc").length;D(".totalPage").text(p);D(".top_Num").show();if(o=="boolean"&&isHasDetailReport==true){m(a?a:1);AddH(a?a:1);d(t,i);size=a?a:1;f(s,isWordDetailReport)}function d(e,t){if(e==""||e=="null"){D(".pageNum").text(1);D(".input_pageNum").val(1)}else{D(".pageNum").text(e);D(".input_pageNum").val(e);D("#toall_Num").attr("href","#"+e);D("body").scrollTop(t)}}function f(e,t){if(t){if(e==""||e=="block"){D(".allButton").hide();D(".CollagenButton").hide();D("#menuButton").css("background","#323639")}else{D(".allButton").show();D(".CollagenButton").show();D("#menuButton").css("background","#1b1e1f")}}else{if(e==""||e=="block"){D(".allButton").hide();D(".CollagenButton").hide();D("#menuButton").css("background","#f2f2f2")}else{D(".allButton").show();D(".CollagenButton").show();D("#menuButton").css("background","#f2f2f2")}}}var g=1;function h(){var a=D("body").scrollTop();var o=-15;D(".upkmzjakmc").each(function(e,t){var i=D(this)[0].getBoundingClientRect().height+15;if(o<a&&a<=o+i){g=e+1}o+=i});D(".pageNum").text(g);D(".input_pageNum").val(g);D("#toall_Num").attr("href","#"+g)}function m(e){_Size=Math.round(e*100)/100;S=Math.round(_Size*100);if(S==""){D(".percent_num").text(100)}else{D(".percent_num").text(S)}D(".upkmzjakmc").css("transform","scale("+_Size+")");if(_Size<1){D(".upkmzjakmc").css("transform-origin","left top")}else{D(".upkmzjakmc").css("transform-origin","center top")}if(g!=undefined){D(".jump_page").attr("href","#"+g)}D(".upkmzjakmc").each(function(e,t){var i=this.getBoundingClientRect().width+30;var a=this.getBoundingClientRect().height+15;D(this).parent(".jcjymywwzd").css("width",i);D(this).parent(".jcjymywwzd").css("height",a)});D("body").off("scroll").on("scroll",function(){h(_Size);var e=D("body").scrollTop();var t=g;var i={PageNum:t,page_nav_click:H};parent.parent.postMessage(i,"*");if(o=="boolean"&&isHasDetailReport==true){if(n){M.operateState.saveDetailNum(Report.report_id,t);M.operateState.saveDetailOffset(Report.report_id,e)}else{M.operateState.savePlainTextDetailNum(Report.report_id,t);M.operateState.savePlainTextDetailOffset(Report.report_id,e)}}});var t=_Size;if(o=="boolean"&&isHasDetailReport==true){if(n){M.operateState.saveDetailSize(Report.report_id,t)}else{M.operateState.savePlainTextDetailSize(Report.report_id,t)}}}D(".toall_div").mousemove(function(){if(n){D(this).css({"background-color":"#424649","border-radius":"2px"})}else{D(this).css({border:"1px solid #dbdbdb","background-color":"#fbfbfb"})}D(this).children(".tip").css("display","block")}).mouseout(function(){if(n){D(this).css("background-color","transparent")}else{D(this).css({"background-color":"transparent",border:"1px solid transparent"})}D(this).children(".tip").css("display","none")});D(".Switch").click(function(){D(".layui-layer-shade").show();D(".layui-layer-title").html(multilingual[l].SwitchMode);D(".detail_btn").data("type","switch");if(n){D(".layui-layer-content").html(multilingual[l].TextMode)}else{D(".layui-layer-content").html(multilingual[l].WordMode)}});D(".detail_btn").click(function(){type=D(this).data("type");if(type=="switch"){D(".layui-layer-btn0").attr("href",D(this).data("txtLink"));if(o=="boolean"&&isHasDetailReport==true){if(n){M.operateState.saveInitDetailReport(Report.report_id,"text");parent.postMessage("switch-detail-text","*");parent.parent.postMessage("switch-detail-text","*")}else{M.operateState.saveInitDetailReport(Report.report_id,"word");parent.postMessage("switch-detail-word","*");parent.parent.postMessage("switch-detail-word","*")}}}if(type=="subline"){var e=D("span").hasClass("red");var t=D("span").hasClass("orange");if(e==false&&t==false){b("")}else{var i=D(".red").parent().eq(0).hasClass("red_a")||D(".orange").parent().eq(0).hasClass("orange_a");var a=String(i);b(a);if(n){M.operateState.saveLineState(Report.report_id,a)}else{M.operateState.savePlainTextLineState(Report.report_id,a)}}}});D(".layui-layer-close, .layui-layer-btn1, .layui-layer-btn0").click(function(){D(".layui-layer-shade").hide()});D(".subline").click(function(){D(".layui-layer-shade").show();D(".layui-layer-title").html(multilingual[l].AuxiliaryLine);D(".detail_btn").data("type","subline");if(D(".red").parent().eq(0).hasClass("red_a")||D(".orange").parent().eq(0).hasClass("orange_a")){D(".layui-layer-content").html(multilingual[l].CancelLine)}else{D(".layui-layer-content").html(multilingual[l].UseLine)}curHref=D(".detail_btn").attr("href");if(curHref!="#"){D(".detail_btn").data("txtLink",D(".detail_btn").attr("href"));D(".layui-layer-btn0").attr("href","#")}});function b(e){if(e==""||e=="true"){D(".red").parent().removeClass("red_a");D(".orange").parent().removeClass("orange_a");if(n){D(".subline").css("background-position","-106px -162px")}else{D(".subline").css("background-position","-161px -9px")}}else{D(".red").parent().addClass("red_a");D(".orange").parent().addClass("orange_a");if(n){D(".subline").css("background-position","-146px -159px")}else{D(".subline").css("background-position","-43px -153px")}}}b(r);D(".bottom").click(function(){D(".bottom_a").attr("href","#"+(g+1));D("#toall_Num").attr("href","#"+(g+1))});D(".top").click(function(){D(".top_a").attr("href","#"+(g-1));D("#toall_Num").attr("href","#"+(g-1))});D(".input_pageNum").focus(function(){if(n){D(".input_pageNum").css("background-color","#424649")}else{D(".input_pageNum").css("background-color","#fbfbfb")}D(".input_pageNum").select()}).blur(function(){D(".input_pageNum").css("background-color","transparent");page1=D(".input_pageNum").val();if(page1<=p&&/^[1-9]\d*$/.test(page1)){D("#toall_Num").attr("href","#"+page1)}else{D(".input_pageNum").val(g)}}).mousemove(function(){if(n){D(".input_pageNum").css("background-color","#424649")}else{D(".input_pageNum").css("background-color","#fbfbfb")}D(this).siblings(".tip").css("display","block")}).mouseout(function(){if(D(".input_pageNum").is(":focus")){if(n){D(".input_pageNum").css("background-color","#424649")}else{D(".input_pageNum").css("background-color","#fbfbfb");D(this).siblings(".tip").css("display","none")}}else{D(".input_pageNum").css("background-color","transparent");D(this).siblings(".tip").css("display","none")}}).keyup(function(){if(event.keyCode==13){page1=D(".input_pageNum").val();if(page1<=p&&/^[1-9]\d*$/.test(page1)){D("#toall_Num").attr("href","#"+page1);window.location.hash="#"+page1}else{D(".input_pageNum").val(g);D(".input_pageNum").blur()}}});var _=M.operateState.getFoldState(Report.report_id);v(_);function v(e){if(e==""||e=="block"){D(".btn_fold_left").css("display","none");D(".btn_fold_right").css("display","block")}else{D(".btn_fold_left").css("display","block");D(".btn_fold_right").css("display","none")}var t={FoldState:e};parent.postMessage(t,"*");parent.parent.postMessage(t,"*");M.operateState.saveFoldState(Report.report_id,e)}D(".btn_fold").click(function(){var e=D(".btn_fold_left").css("display");_size=M.operateState.getDetailSize(Report.report_id);v(e);AddH(_size)});var k=hasScrollbar();if(k){var y=getScrollbarWidth();D(".btn_fold").css("right",y+"px");D("body").css("border-right","0");D(".fixed_shadow").css("width","calc(100% - "+y+"px)")}else{D(".btn_fold").css("right",0+"px");D("body").css("border-right","1px solid #e8e8e8");D(".fixed_shadow").css("width","100%")}D("#originalButton").on("click",function(){D(".zoom-shade").show();size=1;_size=size;m(_size);AddH(_size);w()});D("#fitPageButton").click(function(){D(".zoom-shade").show();var e=D("body")[0].getBoundingClientRect().height;size=(e-20)/minHeight;_size=size;m(_size);AddH(_size);w()});D("#fitWidthButton").click(function(){D(".zoom-shade").show();var e=D("body")[0].getBoundingClientRect().width;size=(e-163)/minWidth;if(size>2){size=2}_size=size;m(_size);AddH(_size);w()});function z(e,t){return Math.round(e*Math.pow(10,t))/Math.pow(10,t)}var S;D("#zoomOutButton").click(function(){D(".zoom-shade").show();if(S%10==0&&size<=2){size+=.1}if(S%10>=5&&size<=2){size=z(S/100,1)}if(S%10<5&&size<=2){size=z(S/100,1)+.1}if(size>2){size=2}_size=size;m(_size);AddH(_size);w()});D("#zoomInButton").click(function(){D(".zoom-shade").show();if(S%10==0&&size>=.1){size-=.1}else if(S%10>=5&&size>=.1){size=z(S/100,1)-.1}else if(S%10<5&&size>=.1){size=z(S/100,1)}if(size<.1){size=.1}_size=size;m(_size);AddH(_size);w()});D(".allButton").mousemove(function(){if(n){D(this).css("background","#1b1e1f")}else{D(this).css("background-image","-webkit-linear-gradient( 90deg, rgb(216,216,216) 0%, rgb(239,239,239) 100%)")}D(this).children(".bot_tip").css("display","block")}).mouseout(function(){if(n){D(this).css("background","#323639")}else{D(this).css("background","#f2f2f2")}D(this).children(".bot_tip").css("display","none")});D("#menuButton").click(function(){var e=D(".allButton").css("display");D(".percent_list").hide();if(n){D("#CollagenButton").css("background","#323639")}else{D("#CollagenButton").css("background","#f2f2f2")}if(o=="boolean"&&isHasDetailReport==true){if(n){M.operateState.saveZoomBtnState(Report.report_id,e)}else{M.operateState.savePlainTextZoomBtnState(Report.report_id,e)}}if(e=="block"){D(".allButton").hide();D(".CollagenButton").hide();if(n){D("#menuButton").css("background","#323639")}else{D("#menuButton").css("background","#f2f2f2")}}else{D(".allButton").show();D(".CollagenButton").show();if(n){D("#menuButton").css("background","#1b1e1f")}}}).mousemove(function(){if(n){D("#menuButton").css("background","#1b1e1f")}D(this).children(".menu_tip").css("display","block")}).mouseout(function(){D(this).children(".menu_tip").css("display","none");if(D(".allButton").css("display")=="block"){if(n){D("#menuButton").css("background","#1b1e1f")}}else{if(n){D("#menuButton").css("background","#323639")}else{D("#menuButton").css("background","#f2f2f2")}}});D("#CollagenButton").click(function(e){var t=D(".percent_list").css("display");if(t=="block"){D(".percent_list").hide();D(this).children(".percent_tip").css("display","block")}else{D(".percent_list").show();if(n){D("#CollagenButton").css("background","#1b1e1f")}else{D("#CollagenButton").css("background-image","-webkit-linear-gradient( 90deg, rgb(224,224,224) 0%, rgb(170,170,170) 170%)")}}D(this).children(".percent_tip").css("display","none");e.stopPropagation()}).mousemove(function(){if(n){D("#CollagenButton").css("background","#1b1e1f")}else{D("#CollagenButton").css("background-image","-webkit-linear-gradient( 90deg, rgb(224,224,224) 0%, rgb(170,170,170) 170%)")}if(D(".percent_list").css("display")=="block"){D(this).children(".percent_tip").css("display","none")}else{D(this).children(".percent_tip").css("display","block")}}).mouseout(function(){if(D(".percent_list").css("display")=="block"){if(n){D("#CollagenButton").css("background","#1b1e1f")}else{D("#CollagenButton").css("background-image","-webkit-linear-gradient( 90deg, rgb(224,224,224) 0%, rgb(170,170,170) 170%)")}}else{if(n){D("#CollagenButton").css("background","#323639")}else{D("#CollagenButton").css("background","#f2f2f2")}}D(this).children(".percent_tip").css("display","none")});D(document).click(function(){D(".percent_list").hide();if(n){D("#CollagenButton").css("background","#323639")}else{D("#CollagenButton").css("background","#f2f2f2")}});D(".zoom-shade").click(function(e){var t=D(".percent_list").css("display");if(t=="block"){D(".percent_list").show()}else{D(".percent_list").hide()}e.stopPropagation()});D(".percent_li").click(function(e){D(".zoom-shade").show();var t=D(this).children(".li_text").text();size=t/100;_size=size;m(_size);AddH(_size);w();e.stopPropagation()}).mousemove(function(){if(n){D(this).css("background","#1b1e1f")}else{D(this).css("background","#e9e9e9")}}).mouseout(function(){if(n){D(this).css("background","#323639")}else{D(this).css("background","#f2f2f2")}});function w(){setTimeout(function(){D(".zoom-shade").hide()},200)}function x(e){return JSON.stringify(JSON.parse(e)).replace(/</g,"&lt;").replace(/>/g,"&gt;")}var B=JSON.parse(x(detail_title_info));if(o=="boolean"&&isHasDetailReport==true){D("#mbxsgxktna a[target='right']").mouseover(function(){var e=`<div class="similarTip"><div>${multilingual[l].Similarity}: <span class="num"></span>%</div></div>`;var t=`<div>${multilingual[l].SimilarityTip} !</div>`;D("#mbxsgxktna").append(e);var i=D(this).data().id;D(this).parent().css("z-index","");var a=M.operateState.getFoldState(Report.report_id);if(a=="none"){D(".similarTip").append(t)}D("div[class='similarTip'] .num").html(B[i].score)}).mouseout(function(){D("div[class='similarTip']").remove()}).mousemove(function(e){var t=e||window.event;var i=document.documentElement.clientWidth;var a=document.documentElement.clientHeight;var o=t.clientY+20;var n=t.clientX+20;if(i-t.clientX<120){n=t.clientX-100}if(a-t.clientY<70){o=t.clientY-50}D("div[class='similarTip']").css({left:n+"px",top:o+"px"})}).click(function(){var e=M.operateState.getFoldState(Report.report_id);if(e=="none"){return false}})}var R=document.querySelector("#mbxsgxktna").cloneNode(true).innerHTML;var C=document.querySelector("style").cloneNode(true).innerHTML;var N={sheetCSS:C,bodyDOM:R};parent.parent.postMessage(N,"*");D(document).on("click",".edit_icon",function(){var e=D(this).prev().text();var t={section_edit:e};parent.parent.postMessage(t,"*")});D(document).on("click",".modify_document",function(){var e={modify_document:true};parent.parent.postMessage(e,"*")});var H="false";window.addEventListener("message",function(e){if(e.data.EnterId){window.location.hash="#"+e.data.EnterId}if(e.data.page_nav_click){H=e.data.page_nav_click}if(e.data.leng){l=e.data.leng;c(l);M.operateState.saveLanguage(Report.report_id,l)}});parent.parent.postMessage("Page_Loading","*")})})(Report,jQuery);function hasScrollbar(){return document.body.scrollHeight>(window.innerHeight||document.documentElement.clientHeight)}function getScrollbarWidth(){var e=document.createElement("div");e.style.cssText="width: 99px; height: 99px; overflow: scroll; position: absolute; top: -9999px;";document.body.appendChild(e);var t=e.offsetWidth-e.clientWidth;document.body.removeChild(e);return t}var arrW=new Array;var arrH=new Array;$(".upkmzjakmc").each(function(e,t){arrW.push($(this).width());arrH.push($(this).height())});function getMaxCountWidth(e){var t=new Array;for(var i in e){var a=e[i];var o=0;if(!t.hasOwnProperty(a)){t[a]=1}else{o=t[a];o=o+1;t[a]=o}}var n=0;var s=0;for(var r in t){var l=t[r];if(l>=n){n=l;if(s==0){s=r}else if(r>s){s=r}}}return s}function getMaxCountHeight(e){var t=new Array;for(var i in e){var a=e[i];var o=0;if(!t.hasOwnProperty(a)){t[a]=1}else{o=t[a];o=o+1;t[a]=o}}var n=0;var s=0;for(var r in t){var l=t[r];if(l>=n){n=l;if(s==0){s=r}else if(r<s){s=r}}}return s}var minWidth=getMaxCountWidth(arrW);var minHeight=getMaxCountHeight(arrH);var size=1;var _size;function AddH(e){var t=document.documentElement.clientWidth;var i=document.documentElement.clientHeight;var a=$(".upkmzjakmc:last")[0].getBoundingClientRect().height+30;var o=i-a;if(a>=i){$("#Add_height").css("height",0)}else{$("#Add_height").css("height",o)}var n=minWidth*e;var s=$(".top_Num").width()+25;var r=(t-n)/2;if(n<=500){r=r-s}else if(n>=t-105){r=77}else{r=r+15}$(".top_Num").css("right",r)}$(window).resize(function(){AddH(size);var e=hasScrollbar();if(e){var t=getScrollbarWidth();$(".btn_fold").css("right",t+"px");$("body").css("border-right","0")}else{$(".btn_fold").css("right",0+"px");$("body").css("border-right","1px solid #e8e8e8")}});