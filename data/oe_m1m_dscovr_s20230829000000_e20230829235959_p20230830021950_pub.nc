CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230829000000_e20230829235959_p20230830021950_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-08-30T02:19:50.514Z   date_calibration_data_updated         2023-08-08T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-08-29T00:00:00.000Z   time_coverage_end         2023-08-29T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx��p�  �          @�33?�  �x��@�(�B^�C�8R?�  � ��@ҏ\B�L�C���                                    Bx��&  �          @�
=?����a�@��Bh{C�e?����
=q@�G�B��=C�!H                                    Bx���  �          @�  ?�
=�W�@�z�Bd
=C�� ?�
=��
@�\)B�=qC�ff                                    Bx��r  �          @�?�
=���@��BF�C�Ф?�
=�]p�@���Bv��C�e                                    Bx��  T          @���?�����{@�{BRC���?����>�R@�Q�B��C�                                      Bx�鹾  �          @���?�Q����R@��BBQ�C�)?�Q��c�
@ٙ�Br(�C��                                    Bx���d  �          @�Q�?�����@�z�B:�C�J=?���s33@�z�Bk  C���                                    Bx���
  �          @���?�ff���@�B4��C�]q?�ff�u@�Bd��C��q                                    Bx���  �          @��?�����@�(�B33C��
?������@�G�BH33C�N                                    Bx���V  �          @��?����ʏ\@w
=A�  C��?������\@��B-�C���                                    Bx���  �          @�?�{��=q@x��A�=qC�R?�{���@�B-��C�5�                                    Bx���  �          @�?��H��p�@��\B{C���?��H���
@��HB4\)C��
                                    Bx�� H  �          @�\?=p���p�@}p�A��C�L�?=p���z�@���B0�C���                                    Bx��.�  �          @��H?��ҏ\@o\)A�(�C�L�?���33@�33B(�HC��3                                    Bx��=�  �          @��
>�����@l(�A癚C���>���p�@�=qB&�C�Q�                                    Bx��L:  �          @��H?�����@g
=A�
=C�\)?����{@��B$p�C�                                    Bx��Z�  
�          @�?(��ҏ\@i��A�33C���?(����@�Q�B&z�C��                                    Bx��i�  T          @���?:�H�У�@l��A�G�C�4{?:�H����@���B(�C���                                    Bx��x,  �          @���?   ��p�@]p�A��HC�  ?   ���@��B �\C�xR                                    Bx���  �          @�?fff��z�@s33A�C��?fff��z�@��
B,��C���                                    Bx��x  �          @�ff?����  @���B�C�*=?������@�Q�B>�C�J=                                    Bx��  �          @��?����ff@���B=qC�33?�����H@�  B?�C�Y�                                    Bx���  �          @���?aG�����@���BC�q?aG���
=@���B;�C��q                                    Bx���j  �          @�?��\@��B�C���?���Q�@��
B;=qC�>�                                    Bx���  �          @�\?:�H��G�@��HB��C�s3?:�H��
=@��\B:�
C�/\                                    Bx��޶  �          @��?h�����@�ffB{C�k�?h������@���B@{C�g�                                    Bx���\  �          @�  ?W
=���R@�z�B{C�33?W
=���\@��BH33C�7
                                    Bx���  �          @�G�?0������@�\)BC�Z�?0������@�{BA33C�q                                    Bx��
�            @�G�?L������@��RB+��C�O\?L������@���B^C��q                                    Bx��N  
�          @�׾����33@<(�A�33C�G��������@���Bz�C�
=                                    Bx��'�  �          @�  �u���@AG�A��C��R�u���R@�\)B��C��3                                    Bx��6�  �          @�ff�L������@Tz�A�ffC�&f�L����\)@��BG�C��                                    Bx��E@  T          @��H�B�\��Q�@u�A�C�q�B�\��\)@���B2{C��                                    Bx��S�  �          @�  =�\)���R@�ffB�
C�XR=�\)����@�(�BL{C�o\                                    Bx��b�  �          @��>�{��\)@�\)B(\)C��>�{����@�G�B\�\C�s3                                    Bx��q2  �          @�(�=�G���G�@��B��C��
=�G���(�@��
BP\)C���                                    Bx���  �          @�R<#�
��=q@�  BffC��<#�
���R@��RBD�HC��                                    Bx��~  �          @�{=#�
��
=@��B�C�33=#�
���\@�G�BI�C�>�                                    Bx��$  �          @�(�>#�
��33@�p�B{C���>#�
��{@��\BM�RC��                                    Bx���  �          @�33�L������@�G�B��C���L������@�
=BIz�C��=                                    Bx��p  �          @��
������@��B  C�\)����@��
BCC�5�                                    Bx���  �          @�{�.{���\@�ffB33C�.�.{���R@�p�BD
=C��q                                    Bx��׼  �          @�p���\)��Q�@�Q�B��C�� ��\)��(�@��RBF��C�J=                                    Bx���b  �          @���
=��{@�Q�B�
C��þ�
=���@���B<�C���                                    Bx���  �          @�
=�@  ��(�@n�RA�33C��f�@  ���@�G�B1=qC��
                                    Bx���  �          @�=q�c�
��{@q�A��C���c�
���@�33B1{C��                                    Bx��T  �          @�=q��p��Ϯ@G�A�Q�C�����p���33@���B\)C��                                    Bx�� �  �          @����θR@3�
A���C�����z�@�
=B��C}��                                    Bx��/�  �          @�G���  �љ�@5�A��\C�����  ��
=@���B(�CW
                                    Bx��>F  �          @�G����R��33@.{A��C��q���R��G�@�p�BC�                                    Bx��L�  �          @�
=�Ǯ��{@	��A��C�33�Ǯ����@e�A�C~�H                                    Bx��[�  �          @���\)�У�@!�A��C~���\)��  @~{B{C|�                                    Bx��j8  �          @��a���\)?��AQ�Cm5��a����\@�
A���Ck8R                                    Bx��x�  �          @ə��j=q��  ?�ffA=qCi�{�j=q���
@(�A���Cg��                                    Bx�쇄  �          @Ǯ��{��=q?333@�ffCc���{����?�  A�{Ca�f                                    Bx��*  �          @��H���\��  ?#�
@�
=Cc�\���\��\)?�A}��Cb                                    Bx���  �          @��׿@  ��p�@�A�p�C��H�@  ���@K�B
(�C�T{                                    Bx��v  �          @��?������\@J�HA�=qC��3?�����ff@���B0  C�3                                    Bx���  �          @�ff?����
=@`  B
33C���?���qG�@�Q�B;��C�9�                                    Bx����  T          @�  @Q���  @]p�B�\C��@Q��s33@�\)B7(�C��)                                    Bx���h  �          @��þ�z����
@#�
A݅C�K���z���z�@h��B%
=C���                                    Bx���  �          @�G���p���?�@�  CdB���p���p�?�{Al��Cb��                                    Bx����  �          @�\)�����u�>��?��CZ�����l(�?�ffA�CX��                                    Bx��Z  �          @�Q��z�H����?p��A
{Cf���z�H���@�\A�Q�Cd^�                                    Bx��   �          @�p��������H?=p�@��Cd�R������G�?�A�z�Cb�                                     Bx��(�  �          @�ff�k���33?Q�@���Ch���k�����?�Q�A���Cf�
                                    Bx��7L  �          @����fff���?+�@�G�Ci�R�fff���?���A�  Cg�q                                    Bx��E�  T          @��H�_\)��33?���A"=qCjE�_\)���R@(�A�G�Cg�3                                    Bx��T�  �          @���O\)��\)?˅Ao
=Cl�q�O\)���R@.�RAҏ\Cj
=                                    Bx��c>  �          @ƸR�A�����?�33Av�RCo��A����@5�A�33Cl��                                    Bx��q�  �          @�=q�G���p�@p�A��Cn�)�G���@g�B�Cj��                                    Bx�퀊  �          @�=q�$z���  ?��AiCu޸�$z���{@>{A��Cs�)                                    Bx��0  �          @��
�6ff��p�?�ffA|��CsQ��6ff���\@FffA��\Cp�                                    Bx���  �          @��H�8����
=?�
=A&�\Cs8R�8������@!G�A�CqG�                                    Bx���|  �          @ҏ\��z����@@��A�  C�  ��z�����@���B"C}�                                    Bx���"  �          @�G������p�@3�
A˅C}�
������\@��HB  C{33                                    Bx����  �          @����\)���R@p�A��\Cx���\)��Q�@aG�B�
Cu�                                    Bx���n  �          @љ��8Q�����?��A��\Cr}q�8Q����@J=qA��Co�)                                    Bx���  �          @��H�+���(�@n�RB
�C�xR�+����@�p�BA��C��q                                    Bx����  T          @��
�xQ���{@eB��C��3�xQ����@���B;
=C���                                    Bx��`  �          @ҏ\��33��  @VffA�{C����33����@��\B1�HC��                                    Bx��  �          @��Ϳ�  ��\)@\(�A��C~����  ��\)@�p�B2�C{s3                                    Bx��!�  �          @�G���ff���\@y��B\)C����ff�~�R@���BH��C{�                                    Bx��0R  �          @�=q��
=���H@a�B{C��
=��=q@�
=B8G�C{��                                    Bx��>�  �          @�G�������G�@W
=A�{C{��������@��B1  Cw{                                    Bx��M�  �          @�  ��=q��p�@L(�A�  C}���=q��
=@�p�B,{Cz��                                    Bx��\D  �          @�ff��  ����@G�A�z�C��῀  ���H@�(�B,  C��
                                    Bx��j�  �          @��
���R��z�@+�A�\)C�˅���R���@\)B{C��                                    Bx��y�  �          @θR��\)��  @%�A���C�O\��\)��{@z�HB��C~�                                    Bx��6  �          @�
=��(����@G�A�(�C{G���(���(�@Y��A�Cy�                                    Bx���  �          @Ϯ�����@33A��Cz!H���33@Z�HA��\Cw�3                                    Bx�  �          @����   ��(�@�\A�p�Czh��   ��z�@g
=B	\)Cw��                                    Bx��(  �          @��������{@4z�AхC{��������=q@��\BCxO\                                    Bx����  �          @��
����33@��A�
=C{�
�����\@n{B��Cy.                                    Bx���t  �          @�33��\��@(�A���C|���\���R@a�B�CzT{                                    Bx���  �          @�ff�У����
@\��B�C{��У��vff@���B=��Cw�                                    Bx����  �          @����33��p�@c33B�
C����33�hQ�@�33BI��C|c�                                    Bx���f  �          @��H�+����@~{B#(�C��ÿ+��Z�H@��B\Q�C�z�                                    Bx��  T          @�\)��=q��ff@E�A�=qC��)��=q����@�
=B5��C                                      Bx���  �          @��R������@&ffAθRC��������@s�
BG�C|u�                                    Bx��)X  �          @�\)��
=��  @p�A�z�C~�ῷ
=���R@l��B(�C|33                                    Bx��7�  �          @�\)��=q��\)@J�HA��Cx�\��=q�qG�@�  B5��Ct)                                    Bx��F�  �          @�Q쿾�R���
@EA��
C|����R�{�@�
=B4\)Cy.                                    Bx��UJ  �          @�p�����
=@fffB�RCz�����j=q@�BE(�Cus3                                    Bx��c�  �          @ƸR��(���
=@Z�HB�\C}�쿼(��|(�@��B=\)Cy�)                                    Bx��r�  �          @��Ϳ�p�����@\(�B	{C�׿�p��w�@��\BA�RC|Q�                                    Bx��<  �          @��
�����(�@A�A�33C~�f�����{@�\)B/�C{�R                                    Bx���  "          @��
�������@g�B�HC{������dz�@�{BI33Cv��                                    Bx�  �          @�ff��
��z�@a�Bz�Cv
=��
�e�@�33B@\)Cp{                                    Bx��.  �          @�������@I��A��Cr�H����mp�@��B/\)Cm.                                    Bx���  �          @�
=��
=��G�@Y��B=qC�lͿ�
=��  @��\B>C}��                                    Bx���z  �          @��ÿ�����@K�A���C}T{�����p�@���B2�HCy��                                    Bx���   �          @ȣ׿�Q����H@(�A���C����Q�����@s33BG�C}E                                    Bx����  �          @ƸR�����H@<(�A��HCzc׿����@��B*
=Cvp�                                    Bx���l  �          @ə���ff��  @FffA�{C}���ff����@�33B.��Cz
                                    Bx��  �          @ə�������@0  AϮC{������
=@���B 33Cx33                                    Bx���  �          @�녿�=q����@9��Aڏ\Cz�q��=q���
@�p�B%�CwO\                                    Bx��"^  �          @˅�����G�@>�RAߙ�CzJ=������\@�  B'�
CvaH                                    Bx��1  �          @�G���z����@1�A�{Cz+���z�����@�=qB!(�Cv}q                                    Bx��?�  �          @ȣ�������@%�A���Cw������@w
=B�Cs�R                                    Bx��NP  
*          @�\)�$z�����@(Q�A��Cr���$z����@vffB�
CnO\                                    Bx��\�  T          @Ǯ�{��(�@!�A��Ct@ �{��G�@r�\B�Cp�                                    Bx��k�  
�          @�(��)����
=@�A��\Cq��)����@eB�Cm�{                                    Bx��zB  
�          @�G��(�����@(�A���Cv���(���\)@l(�B�Cr��                                    Bx����  
(          @��
�������@1G�A���Cy#׿�����ff@���B#�RCu!H                                    Bx��  
�          @�ff���H���@8��A�Q�Cy
=���H��p�@�(�B'�Ct�
                                    Bx��4  "          @\�5���@A��Cp!H�5���ff@Tz�B��Cl�                                    Bx���  
�          @�\)��������@A�=qC}޸������\)@n{BC{B�                                    Bx��À  
�          @�{�(�����
@ ��A��Cr�(�����@R�\B��Co)                                    Bx���&  �          @Å�s33��ff?�  A���Ce���s33�u�@7�A�  Ca0�                                    Bx����  "          @�z��|������?�z�A�\)CcG��|���g�@>�RA��HC^u�                                    Bx���r  
Z          @�z��e���?\Ae��Ch���e���@.�RA�G�Ce&f                                    Bx���  "          @��@  ��(�?�33Aw�
Co���@  ��  @=p�A�\)ClL�                                    Bx���  �          @�\)�B�\���?�Q�A{�
Co� �B�\����@@��A�G�Cl{                                    Bx��d  $          @�z��>�R��33?У�Av�\Co�>�R��\)@<(�A�33Clh�                                    Bx��*
  
�          @�ff�7����\?��\A=�Cq���7���G�@*=qA˙�Co�                                    Bx��8�  �          @�{�K���\)?h��A�HCn���K���G�@�A��Cln                                    Bx��GV  T          @���_\)����?=p�@��
Ck+��_\)��(�@�
A���Ci                                      Bx��U�  �          @�(��mp�����?W
=@�p�ChY��mp�����@
=A�\)Ce޸                                    Bx��d�  �          @�  �x����\)?�z�A��HCcn�x���hQ�@0  Aڏ\C^�q                                    Bx��sH  �          @�=q��(����?˅Ar�RCa5���(��dz�@*=qA�C\�)                                    Bx���  
�          @�ff��Q���p�?+�@�C^�q��Q��u�?��
A��C\T{                                    Bx��  �          @Å������R�\)���
C_��������H?��\AG�C^�=                                    Bx��:  �          @�����  �|(����
�EC]0���  �xQ�?B�\@��HC\�q                                    Bx���  �          @��H�G
=����@\)A��Cn\)�G
=���@b�\B��Ci�                                     Bx��  �          @�33�J=q��33@A��Cn.�J=q���\@Z=qBCiٚ                                    Bx���,  �          @ʏ\�C�
���R?��A�  Co���C�
��  @O\)A�  CkǮ                                    Bx����  �          @��H�C�
���@�A�p�CoT{�C�
����@W�B ��Ck0�                                    Bx���x  �          @��H�<(���{@	��A��RCpz��<(���z�@`  B  Cl=q                                    Bx���  
�          @˅�>{��ff@�A�  CpL��>{���@^{B�Cl
                                    Bx���  
�          @��
�P����(�?��A�Q�Cm���P������@O\)A�\Cip�                                    Bx��j  �          @˅�HQ����
@Q�A��\Cn�=�HQ���=q@^{B
=Cj�                                    Bx��#  �          @�(��C�
����?�{A�p�Co��C�
����@P  A�G�Cl#�                                    Bx��1�  T          @��
�Mp���Q�?��An�\Cn���Mp����H@B�\A��
Ck�                                    Bx��@\  T          @ə��O\)��(�?��HA{\)Cm��O\)��ff@Dz�A�33Ci�f                                    Bx��O  
�          @ʏ\�XQ���z�?���AICl���XQ�����@0��Aϙ�CiaH                                    Bx��]�  T          @����w
=���?fffA\)Cg���w
=��p�@{A�  Cd��                                    Bx��lN  "          @ƸR�l����{?8Q�@��
Ci#��l����G�@z�A�  Cf��                                    Bx��z�  �          @�{�s�
���H?��@�33Cg�{�s�
��\)?�Q�A�Q�Ce�H                                    Bx��  �          @ƸR�z�H����?��@�33Cf���z�H��{?��A��HCd�\                                    Bx��@  �          @����z=q��z�>�@��CgO\�z=q����?�A�=qCeQ�                                    Bx���  �          @�  �����z�>�Q�@Tz�Cd#������33?�z�Av�HCbJ=                                    Bx��  T          @����Q���=q?&ff@�{Cf=q��Q���{?��RA�p�Cc��                                    Bx���2  �          @�
=?�  ��(�@`��B�C�*=?�  �{�@�=qBC�\C�|)                                    Bx����  �          @�
=?˅���@b�\B
=C��?˅�|(�@��BCz�C��)                                    Bx���~  T          @�ff?��
����@R�\A��C��\?��
��\)@�{B:��C�l�                                    Bx���$  �          @��
?���G�@g�B
�C���?��r�\@��BI�C�J=                                    Bx����  "          @��H?�����33@p  Bz�C��)?����dz�@��BO�
C��                                    Bx��p  �          @�?�Q���Q�@|(�B�
C��?�Q��[�@�z�BUz�C�)                                    Bx��  �          @Ӆ?�ff��z�@c33B�C���?�ff��z�@�ffBD��C�%                                    Bx��*�  �          @�
=>�G����H@P��A��HC�%>�G�����@���B733C���                                    Bx��9b  
Z          @�  ?�=q���@X��B��C�8R?�=q�}p�@�\)BB=qC�Ff                                    Bx��H  �          @ڏ\?�������@o\)B33C�� ?������R@�BF
=C���                                    Bx��V�  "          @ۅ?�����@y��B
ffC��{?���G�@�G�BI��C�8R                                    Bx��eT  T          @߮@33��33@�{Bz�C�%@33�j=q@��BO=qC�3                                    Bx��s�  "          @��
@�
���@\)B�RC�Ff@�
�k�@�G�BJ�C�3                                    Bx��  �          @��@=q��Q�@��\B�RC��{@=q���\@�Q�BEG�C�P�                                    Bx��F  �          @�R?�p���(�@�z�B��C�N?�p����@�BFffC��                                    Bx���  T          @�\@�
��  @��BffC��)@�
���@��BEp�C�Ǯ                                    Bx��  "          @�R?�p���(�@�(�B
=C��?�p���p�@�33BLffC�<)                                    Bx��8  
�          @�(�?�p�����@~{B��C��f?�p���\)@��RBH(�C�!H                                    Bx����  �          @��?�=q��  @�z�B$G�C��?�=q�Z=q@��Bc
=C�                                      Bx��ڄ  "          @�?����R@��B�C��?��hQ�@�z�B\��C���                                    Bx���*  
X          @���?�(����@�ffB
=C�S3?�(��k�@���BX�C��                                    Bx����  
�          @�33@���z�@�
=B�
C�@��e�@�G�BZQ�C���                                    Bx��v  �          @�z�?����{@��
B{C��R?���tz�@���B[
=C�.                                    Bx��  "          @�  ?�z���33@��B��C�q�?�z��~{@��BW�C��                                     Bx��#�  �          @�  @����@�{B�
C�#�@��u@��
BZp�C��                                    Bx��2h  
�          @�@�����@��BG�C�,�@���n{@���BY��C�P�                                    Bx��A  �          @�@����z�@��BG�C��R@���o\)@��BZG�C��                                    Bx��O�  �          @��@���{@�  BffC�R@��a�@��HB\�C��)                                    Bx��^Z  �          @�R@��\)@���B�HC�\@�c�
@��
B\�\C��                                    Bx��m   "          @�z�@
=����@�33B#��C��H@
=�W
=@�z�Ba  C���                                    Bx��{�  �          @�33@5���
=@�33B%p�C�z�@5��B�\@���B^
=C�~�                                    Bx��L  
�          @��@@����{@�
=B+Q�C��@@���.�R@\Ba{C��f                                    Bx����  �          @�{@$z���G�@��HB"=qC���@$z��U�@�z�B^=qC��\                                    Bx����  "          @�@Q����H@���B
=C��@Q��h��@�p�B[ffC��)                                    Bx���>  "          @��@p���p�@��B(�C�4{@p��p  @��HBV�C��f                                    Bx����  R          @�Q�@"�\��33@�p�B
=C���@"�\�j�H@\BV�HC�T{                                    Bx��ӊ  
�          @��H@@������@���B#�\C�o\@@���>{@��B[G�C��3                                    Bx���0  �          @�ff@G���(�@���B&�C���@G��,��@�z�B[�
C���                                    Bx����  �          @�ff@G���33@��B'��C���@G��*=q@��B]  C���                                    Bx���|  
�          @�p�@]p����@�
=B%(�C��@]p��p�@�  BVQ�C�Ff                                    Bx��"  �          @��@*=q���H@�\)B"\)C�b�@*=q�I��@�  B]��C��                                    Bx���  "          @�\)@\)��G�@�G�B&p�C��)@\)�E�@�G�Bc
=C�~�                                    Bx��+n  �          @�p�@S�
��=q@�p�B"z�C��R@S�
�*�H@���BV��C���                                    Bx��:  �          @�{@I����
=@��B!z�C��R@I���3�
@��BW�C�%                                    Bx��H�  �          @�z�@N{���@���B"��C�33@N{�,��@���BW�C��)                                    Bx��W`  $          @�
=@J=q��G�@���B 33C�ff@J=q�7�@��\BW=qC��)                                    Bx��f  �          @�33@K���z�@��\B ��C���@K��/\)@�
=BW  C���                                    Bx��t�  �          @�33@H�����@�Q�B
=C�~�@H���6ff@�{BU33C��H                                    Bx���R  
�          @�ff@333���H@���B�RC��@333�K�@��BWp�C���                                    Bx����  
�          @�{@8�����@�=qB��C��\@8���Dz�@�=qBXffC��                                    Bx����  T          @�@G����R@���B$�HC�xR@G��/\)@�B\{C�\)                                    Bx���D  �          @�ff@E���
=@��B$ffC�B�@E��0  @���B\�C�{                                    Bx����  T          @�\@K���\)@�33B%��C��=@K��.{@���B\C��{                                    Bx��̐  T          @�
=@Q����H@�(�B#Q�C��f@Q��4z�@\BZQ�C��\                                    Bx���6  T          @陚@P  ���@�ffB   C��
@P  �5@���BW\)C�h�                                    Bx����  
�          @��@S�
����@�{BC��@S�
�333@�z�BV�C�޸                                    Bx����  
Z          @陚@L(����@���B
=C��)@L(��C33@�=qBS=qC�+�                                    Bx��(  "          @�(�@K����
@�G�B��C���@K��J�H@�33BR  C���                                    Bx���  
(          @��H@E��@�
=B(�C��@E�O\)@��BQp�C��\                                    Bx��$t  
(          @�Q�@U�����@��B��C��@U��J�H@�BQ  C�8R                                    Bx��3  "          @�{@Z=q���
@��RBz�C�5�@Z=q�7�@�ffBT��C��R                                    Bx��A�  
�          @�p�@l(���  @��\B"�\C�}q@l(��{@�{BT�
C�R                                    Bx��Pf  �          @�@p�����@�ffB��C�b�@p���'�@�33BO�RC��{                                    Bx��_  "          @�Q�@s33����@�(�B\)C���@s33�333@�33BLz�C���                                    Bx��m�  T          @��@|(���
=@��\BG�C��
@|(��0  @���BI{C���                                    Bx��|X  T          @��H@�  ��(�@�\)BG�C�/\@�  �'�@���BK�HC�ff                                    Bx����  
�          @�(�@�G���G�@�Q�B$�C���@�G��(�@���BR��C���                                    Bx����  
�          @�@_\)����@��B��C��@_\)�;�@�{BV�C��                                    Bx���J  
�          @��R@]p���(�@�=qB�C�]q@]p��k�@��\BDz�C���                                    Bx����  
(          @�Q�@Tz���Q�@��HB��C��=@Tz��r�\@���BE�C���                                    Bx��Ŗ  
�          @��@Mp���ff@��RB�RC�@ @Mp��l(�@�  BK�C��                                     Bx���<  
�          @���@Q���z�@�G�B�HC��@Q��g
=@���BL��C��                                    Bx����  
�          @���@S�
��33@�G�B=qC�� @S�
�dz�@���BL�C�q�                                    Bx���  �          @��H@R�\���@��B
=C�J=@R�\�X��@�BM��C�{                                    Bx�� .  
�          @�G�@R�\��
=@��\B33C�q@R�\�`  @��BI�\C���                                    Bx���  �          @���@?\)��z�@�=qBG�C�� @?\)�j=q@��BL�\C��q                                    Bx��z  �          @�R@7
=��p�@���Bz�C�q�@7
=�XQ�@�Q�BV��C�)                                    Bx��,   �          @�{@C33��\)@���B
=C��f@C33�J=q@���BWC��                                    Bx��:�  
�          @�@X����\)@�{BG�C��\@X���:=q@��BU(�C���                                    Bx��Il  "          @�\@j�H��(�@�Q�BG�C�5�@j�H�1�@���BR\)C�j=                                    Bx��X  �          @�@}p��u�@�=qB*z�C��)@}p���@���BW��C��                                    Bx��f�  
�          @�@}p��u@�33B%p�C���@}p��   @�33BSz�C��q                                    Bx��u^  "          @�@�G���
=@��\B5��C��\@�G�>�33@�Q�B>z�@�{                                    Bx���  �          @�(�@�Q��@�G�B4  C��)@�Q�?�Q�@�B.�HAU�                                    Bx����  "          @�p�@��H���@���B1��C���@��H?�\)@�B-��AF=q                                    Bx���P  
�          @�
=@�33��  @�ffB.\)C�o\@�33?8Q�@�  B0�A ��                                    Bx����  �          @��
@�z���@�B0G�C��=@�zᾙ��@��HBCC�&f                                    Bx����  "          @�p�@���o\)@�\)B(=qC��\@�녿�@�ffBT�C���                                    Bx���B  T          @��H@���Z=q@��HB(z�C�e@����p�@�ffBN�C��q                                    Bx����  
�          @�@�  �E@��B'  C�y�@�  ��Q�@��BGp�C��{                                    Bx���  �          @�\)@�
=�g�@��B'�C��H@�
=��33@\BO��C��H                                    Bx���4  �          @���@�33�s33@�ffB'��C�k�@�33��ff@�BR�
C��)                                    Bx���  "          @���@��
�|(�@��HB#��C��q@��
���H@�(�BP
=C��                                    Bx���  T          @���@�����  @�\)BC�޸@�����
@���BL�HC�z�                                    Bx��%&  T          @�
=@�Q����H@�\)B �\C��@�Q����@\BO�C���                                    Bx��3�  "          @���@�ff��z�@�z�B\)C�ٚ@�ff�"�\@��BHQ�C�n                                    Bx��Br  
�          @�z�@����\)@��B{C�Q�@���;�@���BD33C�)                                    Bx��Q  T          @�@��\���@�=qB
�C�Z�@��\�=p�@�{BA��C�H                                    Bx��_�  �          @�33@tz���\)@\)A�33C��@tz��a�@���B;�C��R                                    Bx��nd  
�          @�@g
=��@k�A뙚C��@g
=�u�@��HB5��C���                                    Bx��}
  �          @�\@U���Q�@eA�RC��@U����@��B4�C�T{                                    Bx����  "          @���@�z���(�@�z�B�C��
@�z��=q@�33BOffC���                                    Bx���V  "          @��@u����@�ffB�RC�J=@u��7�@\BO�C���                                    Bx����  �          @�\)@n�R��=q@�\)B�RC�.@n�R�L(�@�
=BJ�C���                                    Bx����  "          @�
=@s33����@��Bz�C��@s33�_\)@��B@(�C���                                    Bx���H  
�          @�p�@s33��=q@\)A��C��H@s33�e@�33B;�HC�N                                    Bx����  T          @�p�@n�R���@�z�BC�/\@n�R�L��@�(�BH�\C���                                    Bx���  �          @�ff@p�����@��RB��C���@p���6ff@�33BQ�C�ff                                    Bx���:  T          @�{@~�R��p�@��
B\)C���@~�R��@��
BR�HC�N                                    Bx�� �  �          @�p�@S33��{@�Q�BQ�C���@S33�e@���BI�HC�G�                                    Bx���  �          @�z�@N�R��=q@�(�B�\C��@N�R�[�@�\)BN�C��f                                    Bx��,  "          @���@K���Q�@�Q�B33C��@K��i��@�BK�C��\                                    Bx��,�  �          @��H@g���{@��RB(�C��@g��B�\@�p�BMz�C��)                                    Bx��;x  "          @�@����{@�ffB�HC�>�@���$z�@�\)BH
=C��R                                    Bx��J  �          @�@W���33@�G�B\)C��)@W��I��@��BR�RC�u�                                    Bx��X�  
�          @��@b�\��\)@�BC��3@b�\�Dz�@��BNQ�C��H                                    Bx��gj  
�          @�=q@����{@S33A�
=C�n@�����H@�(�B5z�C�9�                                    Bx��v  T          @��@$z���{@XQ�A�=qC�C�@$z����\@�(�B6C���                                    Bx����  T          @���@Fff��Q�@z�HA�p�C�*=@Fff�~{@�
=BBz�C�                                      Bx���\  T          @�{@qG���=q@�Q�B�RC��@qG��8Q�@�{BM��C�W
                                    Bx���  
�          @�Q�@u���H@��B
�RC���@u�J=q@�{BHQ�C�@                                     Bx����  �          @�
=@�p���
=@��B'G�C�P�@�p���@�
=BY��C��                                    Bx���N  
�          @�
=@�  ����@��RB#�C��q@�  �@ϮBY�C���                                    Bx����  �          @�p�@�������@��\B��C��\@����Q�@��
BVQ�C���                                    Bx��ܚ  T          @��\@^�R����@�\)B(�C��@^�R�E�@���BU��C�:�                                    Bx���@  �          @�G�@p  ��33@�G�B�HC��
@p  �2�\@�
=BT\)C��\                                    Bx����  �          @�33@b�\��\)@���BQ�C��@b�\�L(�@�\)BRffC��                                    Bx���  �          @��
@fff����@�=qBG�C�&f@fff�P��@�BOffC��{                                    Bx��2  
�          @��\@c�
��33@�
=B=qC��@c�
�Vff@ÅBMffC�XR                                    Bx��%�  T          @�
=@tz�����@���B
=C��@tz��Q�@���BJ��C��                                    Bx��4~  
�          A (�@��H����@��RBQ�C��=@��H���@ȣ�BN��C��\                                    Bx��C$  
�          A ��@�ff���
@�=qB{C�h�@�ff��(�@�
=BJ��C��H                                    Bx��Q�  �          A z�@�{���
@���B  C�Y�@�{��(�@ƸRBJC��{                                    Bx��`p  T          A ��@�����Q�@��
B��C�W
@�����@ʏ\BOG�C��                                    Bx��o  T          AG�@�����p�@���B�C���@������@ə�BL��C��H                                    Bx��}�  T          Ap�@�  ����@�Bz�C�7
@�  ��
@���BQ�C���                                    Bx���b  
�          A Q�@�p��Tz�@�z�B(�HC��q@�p����@�\)BL(�C��q                                    Bx���  T          A z�@�  �s33@�{B*�HC���@�  ��G�@θRBV�RC��{                                    Bx����  �          A Q�@�Q�����@�  B#�\C�@�Q��@�(�BR�C�3                                    Bx���T  "          @�\)@�����p�@�(�B)��C�p�@��Ϳ�\)@�G�B\�RC��                                    Bx����  �          @��@tz���=q@��\B+{C��f@tz��G�@��BbffC�R                                    Bx��ՠ  	�          @��H@e����R@���B�C��=@e��1G�@���BZ�C�"�                                    Bx���F  �          @��@8Q���z�@�=qA��C��3@8Q���@�=qBHp�C�P�                                    Bx����  T          @�@1G���  @���A�C���@1G���G�@�=qBG��C�o\                                    Bx���  �          @�p�@,������@�  A�RC���@,����=q@�=qBG�HC�H                                    Bx��8  �          @��\@�H���@qG�A�\C�Y�@�H����@���BD  C��                                    Bx���  �          @��H@"�\��p�@g�A�=qC�,�@"�\��33@�BA��C��                                    Bx��-�  �          @�33@G���@W�Aң�C���@G���@���B;\)C��
                                    Bx��<*  
�          @���@)����ff@z�HA�C��{@)����Q�@�
=BG�C���                                    Bx��J�  "          A ��@J=q�Å@�A�{C��\@J=q��=q@�BH�\C��                                    Bx��Yv  "          @���@G����@�=qA�ffC�� @G���  @�G�BGC��
                                    Bx��h  
�          A   @aG����@�ffBQ�C��f@aG��N{@���BUG�C�Ǯ                                    Bx��v�  T          @�\)@\(�����@��B�
C���@\(��@  @�G�B\\)C�k�                                    Bx���h  
�          @�\)@,(���z�@��HB�C�Ф@,(�����@�33BR��C��)                                    Bx���  U          @�
=@AG���33@��A�\)C�0�@AG�����@�p�BJ�RC�`                                     Bx����  �          A (�@;��ȣ�@���A�p�C���@;���Q�@��
BGG�C�AH                                    Bx���Z  �          A ��@333��p�@}p�A뙚C�Ǯ@333��p�@ÅBE��C�.                                    Bx���   T          @�\)@/\)�˅@~{A�\C��=@/\)��33@ÅBG�\C��                                    Bx��Φ  �          @�p�@'���(�@y��A�=qC�*=@'���z�@���BG\)C�k�                                    Bx���L  �          @��@7��\@���A�(�C��@7����@\BJ��C���                                    Bx����  �          @�
=@E��  @���Bp�C��@E�c�
@�p�BV�
C�~�                                    Bx����  �          @���@C33���H@��B��C��f@C33�l��@ȣ�BRp�C��R                                    Bx��	>  �          @�Q�@AG���ff@��B33C���@AG��e�@�BSffC��                                    Bx���  �          @���@?\)����@�G�B��C��=@?\)�j�H@�ffBR�\C���                                    Bx��&�  T          @��@J=q���R@�\)B	��C�}q@J=q�aG�@��HBU
=C��{                                    Bx��50  
(          @�33@P����(�@�(�B�HC�\@P���^�R@�
=BR�C���                                    Bx��C�  �          @�p�@Mp����@���B33C�u�@Mp��k�@�ffBO
=C��f                                    Bx��R|  "          @�
=@:=q�Å@�p�A�
=C���@:=q��  @�
=BN
=C��                                    Bx��a"  
�          @��@Dz����@��B   C��{@Dz��tz�@�z�BM�RC�c�                                    Bx��o�  
)          @�
=@,�����@xQ�A��HC�l�@,����z�@\BF�
C���                                    Bx��~n  T          @�p�@7
=��\)@}p�A�(�C�Q�@7
=��@��HBH��C�.                                    Bx���  
Y          @�p�@5��{@��A��\C�P�@5��33@��BL33C�Y�                                    Bx����  �          @���@>{�\@��\A���C��@>{�~�R@�z�BL(�C�aH                                    Bx���`  "          @�@@  ����@�z�A�{C�0�@@  �{�@�{BMz�C���                                    Bx���  S          @��R@:=q��p�@�33A�  C���@:=q����@�ffBL�RC���                                    Bx��Ǭ  �          @�@1���\)@���A���C��@1���(�@�p�BLQ�C���                                    Bx���R  �          @�p�@   ��z�@~{A�=qC���@   ����@�p�BL�C��                                    Bx����  "          @��@Vff����@xQ�A��HC���@Vff�q�@��BE  C��q                                    Bx���  
Z          @�Q�@\������@w
=A�RC�g�@\���q�@��HBC=qC�5�                                    Bx��D  T          @�Q�@`  ��G�@r�\A�=qC���@`  �s33@���B@�HC�L�                                    Bx���  �          @��@fff��Q�@l��A�z�C�  @fff�s�
@�{B=G�C��\                                    Bx���  
�          @�
=@w����@mp�A噚C�q�@w��g�@��
B:�C�}q                                    Bx��.6  �          @�
=@b�\��33@|��A�Q�C�!H@b�\�c�
@��BE�C�j=                                    Bx��<�  
(          @�p�@XQ���=q@��A�G�C��H@XQ��^�R@�ffBK(�C�{                                    Bx��K�  
�          @�ff@N�R��
=@�Q�A�33C��q@N�R�hQ�@�
=BKG�C���                                    Bx��Z(  �          @�
=@_\)����@��
B C��@_\)�Z=q@��BKQ�C�Ф                                    Bx��h�  �          @���@aG���\)@�Q�B��C�U�@aG��S�
@�33BN�\C�]q                                    Bx��wt  
�          @�ff@Z=q���R@�33B�RC�g�@Z=q�^�R@���BP�C�4{                                    Bx���  T          A ��@p����Q�@���B�HC�+�@p���Mp�@��
BP�RC���                                    Bx����  
�          A33@p������@�  B�RC�"�@p���HQ�@ҏ\BUffC�q                                    Bx���f  
Z          A��@s�
��{@�B	p�C��@s�
�S�
@ҏ\BR
=C���                                    Bx���  T          A\)@q����@��B	=qC���@q��P��@ϮBQ�
C��)                                    Bx����  �          Az�@w���  @���B��C��
@w��Z�H@�
=BM�C�E                                    Bx���X  "          AG�@y������@��B�HC��@y���Z�H@�Q�BM�HC�]q                                    Bx����  �          A{@z�H���@�B��C�z�@z�H�Tz�@ʏ\BK�C��                                     Bx���  "          @��@i�����@�  B=qC��3@i���
=@��RBR��C���                                    Bx���J  �          @��
@qG����\@��
BffC�@qG����@�33BR�\C��                                    Bx��	�  "          @�R@l�����H@�G�B�C���@l���*=q@���BQG�C�)                                    Bx���  
�          @�z�@j=q����@��
B�\C�]q@j=q�HQ�@�p�BKffC��R                                    Bx��'<  �          @���@c33��
=@��\B��C�)@c33�E@��BM{C��                                     Bx��5�  �          @�\@hQ����\@\)A�{C�!H@hQ��N{@�=qBH�C�7
                                    Bx��D�  T          @�{@\(�����@~{B(�C��3@\(��J�H@�G�BLQ�C���                                    Bx��S.  "          @�@U���@�  B  C��@U�L(�@��\BNz�C�                                      Bx��a�  �          @�R@AG���ff@�33BQ�C�~�@AG��QG�@�\)BUG�C�XR                                    Bx��pz  �          @�(�@3�
��ff@��RB�C�  @3�
�\��@�{BY(�C���                                    Bx��   	�          @�@P  ���
@��RBffC��
@P  �I��@���BT�C��{                                    Bx����  
�          @�  @Q���@�Q�B{C���@Q��Q�@���BO33C�}q                                    Bx���l  
�          @�p�@L(�����@���BC���@L(��:=q@���BYffC���                                    Bx���  
�          @�@N{��p�@�=qBQ�C��
@N{�:=q@\BY�
C��R                                    Bx����  �          @�G�@O\)��G�@}p�A��C�"�@O\)�X��@���BN(�C��\                                    Bx���^  "          @��@\������@��B�HC��H@\���HQ�@��BO  C���                                    Bx���  T          @�z�@�33��33@�z�B\)C�aH@�33�3�
@��
BKC���                                    Bx���  
�          @�@�  ��\)@�  B�RC�b�@�  �/\)@�ffBK33C��                                    Bx���P  T          @�\)@w�����@�=qB��C��q@w��1�@�G�BN�
C�+�                                    Bx���  
(          @�(�?��H��p�@n{A�33C�1�?��H���@�p�BN�C���                                    Bx���  �          @�p�@ �����@x��A뙚C���@ ����{@�
=BOz�C�z�                                    Bx�� B  �          @�Q�@ ���ʏ\@n�RA�  C���@ ����p�@���BL�RC�~�                                    Bx��.�  �          @�(�?����G�@\��A���C��?����\)@�(�BI�C�e                                    Bx��=�  "          @�?��ָR@P��A��HC���?����R@�G�BCffC���                                    Bx��L4  
Z          @�Q�@G
=���
@xQ�A��C�z�@G
=�\��@�z�BN�HC���                                    Bx��Z�  �          @��
@3�
���@j�HA�Q�C�n@3�
�|(�@�z�BJ\)C��q                                    Bx��i�  �          @��H@(����@aG�A�
=C���@(�����@��BG�C�]q                                    Bx��x&  �          @���@.�R�ə�@UA��C��{@.�R����@�ffBA33C�:�                                    Bx����  "          @�@'���p�@R�\A�33C�)@'����@��RB@�C�\)                                    Bx���r  �          @�ff@<����33@K�AÙ�C�xR@<����z�@��\B:{C��3                                    Bx���  
�          @��@7���{@P  A��
C�q�@7���
=@��\B>33C�!H                                    Bx����  
�          @�
=@
=��  @]p�Aݙ�C�O\@
=��@���BLffC�aH                                    Bx���d  �          @���@.{����@eA��C�'�@.{�z�H@�=qBJ��C�e                                    Bx���
  T          @�33@ff��G�@a�A�(�C�:�@ff��p�@�(�BKp�C���                                    Bx��ް  
�          @��
@�љ�@Q�AˮC��R@����@���BD�RC�h�                                    Bx���V  �          @��?ٙ�����@Dz�A���C�&f?ٙ���ff@�(�BA��C��                                    Bx����  �          @���?�  ���
@G�A��C�ff?�  ��z�@��BC�C�T{                                    Bx� 
�  �          @�=q?���p�@J=qAŮC�f?����@�
=BD�\C�ٚ                                    Bx� H  "          @�?�z���  @EA��
C��?�z�����@�ffBA�C��)                                    Bx� '�  
(          @�z�?�z���  @J=qA�C��?�z���\)@�Q�BD  C���                                    Bx� 6�  T          @��
?�Q���{@P  A��C�3?�Q���(�@�=qBG  C�H                                    Bx� E:  T          @���?�ff�أ�@EA�Q�C�w
?�ff����@��RB@�C�\)                                    Bx� S�  �          @��?�p��ٙ�@C�
A�ffC�'�?�p����@�ffB@z�C��                                    Bx� b�  �          @��R?���=q@FffA���C���?�����@�  B@�RC���                                    Bx� q,  T          @�
=?�G�����@O\)A�Q�C�G�?�G���ff@��BE�C�B�                                    Bx� �  "          @�z�?�\)�׮@EA��\C���?�\)��\)@��RBA33C���                                    Bx� �x  
�          @�=q?޸R���@J�HA�(�C�N?޸R���@�  BE��C�O\                                    Bx� �  T          @�\?��
��z�@L(�A��C�~�?��
���H@�Q�BF
=C��
                                    Bx� ��  T          @��H?�z��ָR@@  A�(�C��?�z���\)@��
B?=qC���                                    Bx� �j  �          @��H?�p�����@O\)A�C�O\?�p���=q@��BG�
C�e                                    Bx� �  
Z          @�p�?���Q�@y��A�
=C�?��{�@�  B](�C�]q                                    Bx� ׶  
�          @�{@C�
���@�
=B$�C�Q�@C�
�ff@�ffBs�HC�33                                    Bx� �\  �          @��H@Y������@�B��C�k�@Y���+�@У�Ba�C��)                                    Bx� �  �          @��H@+�����@o\)A�RC�H@+��p��@�  BQ��C��R                                    Bx��  "          @�p�@���@Dz�A���C�\@�����@�{B?��C��
                                    Bx�N  T          @�(�?�(��ڏ\@0  A�p�C�?�(���p�@�\)B7��C��                                    Bx� �  "          @�33?�\)���@4z�A�=qC���?�\)���
@�G�B:�RC��                                    Bx�/�  
�          @�z�?�
=��p�@K�Aď\C��?�
=���H@�G�BE\)C�j=                                    Bx�>@  T          @�?�����@P��A��
C��?������@��
BHQ�C�G�                                    Bx�L�  T          @��?�
=��33@>�RA�33C���?�
=���\@�ffB@�C��R                                    Bx�[�  �          @�
=@9����Q�@[�A�G�C�k�@9����33@�33BEffC���                                    Bx�j2  T          @��@<�����@Y��AͮC�]q@<�����@�(�BB��C�n                                    Bx�x�  �          @�G�@���ָR@C�
A�G�C��@�����@�
=B=�C��                                     Bx��~  �          @�33@'���{@HQ�A�=qC���@'���33@���B=�C��3                                    Bx��$  �          @�Q�@p���  @H��A���C�\@p�����@��BA�C��{                                    Bx���  T          @�{?����ۅ@<��A��\C�j=?������\@�ffB?\)C�N                                    Bx��p  �          @�
=?������@7�A���C���?������@���B<ffC���                                    Bx��  	�          @�?��
����@7
=A��HC�8R?��
��z�@�z�B={C��R                                    Bx�м  "          @�\)?˅��  @7
=A�C�b�?˅���@�{B=��C��{                                    Bx��b  
�          @��?��
����@6ffA��HC��?��
��Q�@�ffB=�\C�w
                                    Bx��  �          @�z�?���׮@�A�C�Ǯ?�����R@�z�B0\)C�\)                                    Bx���  "          @��?����ָR@"�\A�z�C��
?�����33@���B5�
C�p�                                    Bx�T  �          @�\)?�����(�@%�A���C���?������@���B8�
C��
                                    Bx��  
�          @�\)?Ǯ���H@%A��C�g�?Ǯ��ff@�z�B8ffC���                                    Bx�(�  
�          @�?�{��\)@#33A��C��H?�{���\@�p�B6  C��                                    Bx�7F  "          @�\)?�=q��@5�A��
C�T{?�=q��=q@�\)B?p�C�f                                    Bx�E�  T          @�?�p��޸R@!G�A�33C�f?�p���=q@�z�B7C���                                    Bx�T�  �          @�R?Ǯ��{@G�A��C�T{?Ǯ���@��B/
=C�e                                    Bx�c8  �          A z�?���G�@4z�A���C�3?���
=@��B9��C���                                    Bx�q�  T          @��\?��ᙚ@8Q�A�Q�C�4{?����@�Q�B=\)C��)                                    Bx���  T          @���?����
=@XQ�Aљ�C��3?�����R@���BQz�C���                                    Bx��*  �          @�\?�
=��(�@XQ�AӮC��?�
=��(�@���BRQ�C�                                    Bx���  "          @�ff?\�У�@P  A���C��3?\���\@��BO�\C��\                                    Bx��v  
�          @�?�
=�Ϯ@G�A���C�>�?�
=��  @�
=B2��C���                                    Bx��  �          @�ff?�����33@!G�A�{C��?�����\)@�  B;\)C�q                                    Bx���  
�          @�?�ff��\)@  A���C�� ?�ff��  @�ffB133C�`                                     Bx��h  
�          @�Q�@����?�Q�A��\C�f@���@���B&33C��R                                    Bx��  
�          @���?�(���z�@�A�z�C�z�?�(����@���B/p�C��                                     Bx���  
�          @���@�
����@�A���C�P�@�
��Q�@��B*\)C���                                    Bx�Z  �          @�z�@ff��(�@�A���C�}q@ff���@��B*=qC��                                    Bx�   
�          @�{@����G�@��A��C���@������@���B4
=C���                                    Bx�!�  �          @���@ff���H?���Ap��C�*=@ff����@���B!{C��                                    Bx�0L  �          @��
@G��Ϯ?���Al��C���@G���@��HB!{C�Z�                                    Bx�>�  
�          @�  @%�ə�?��ALz�C�,�@%��(�@���B�RC��R                                    Bx�M�  �          @���@����?s33@�\)C�u�@������@s33B�RC�Y�                                    Bx�\>  
�          @ۅ@(����ff?n{@�33C��@(����=q@k�B�C���                                    Bx�j�  
�          @���?޸R���?�Ab=qC��3?޸R��p�@�{B!�RC���                                    Bx�y�  "          @�ff?�����33?s33@���C��{?������@x��B�HC�n                                    Bx��0  �          @�=q@z����
���R�"�\C��f@z�����@2�\A���C��=                                    Bx���  �          @��
@*=q�ʏ\?�Q�A|Q�C�e@*=q���R@���B"��C��
                                    Bx��|  �          @��\@*=q�ҏ\@S33A�C�f@*=q����@�\)BG
=C��                                    Bx��"  �          @��@2�\��(�@"�\A��\C���@2�\��
=@��RB2�C��q                                    Bx���  
�          @�@7
=���@Q�A��C�t{@7
=��
=@��B'33C�P�                                    Bx��n  �          @�{@)������@
=qA��C��q@)�����H@�
=B+\)C��
                                    Bx��  T          @�p�@$z���=q@
=A�{C�u�@$z���z�@�{B*C�&f                                    Bx��  �          @�p�@.{����?�Q�A�z�C��@.{��ff@�G�B$
=C���                                    Bx��`  �          @�33@*�H���H?�33A_\)C��)@*�H���@���Bz�C��                                    Bx�  "          @��H@"�\��  ?Q�@�C��@"�\��z�@h��B��C�                                      Bx��  T          @�=q@%�ȣ�?�R@�\)C�5�@%��Q�@^�RA���C��                                    Bx�)R  �          @ָR@=p���\)?�@�G�C�*=@=p���G�@P��A�p�C�7
                                    Bx�7�  "          @��@B�\���>Ǯ@X��C��)@B�\��G�@G
=A��C���                                    Bx�F�  �          @�@<(���
=?   @�G�C��@<(���G�@O\)A�=qC�q                                    Bx�UD  �          @���@N�R�����Ϳ^�RC�L�@N�R��G�@,(�A�C���                                    Bx�c�  
[          @�(�@P���Ǯ���
�p�C��\@P�����@G�A��C�H�                                    Bx�r�  "          @���@Dz���  �h����C��@Dz���{@Q�A��C���                                    Bx��6  
�          @��H@O\)�ə��#�
����C��)@O\)���\@;�A�33C�3                                    Bx���  �          @ᙚ@g���  ����w
=C���@g�����@=qA�Q�C��H                                    Bx���  �          @��@�G����׾.{��33C�~�@�G���p�@#�
A��HC���                                    Bx��(  �          @�=q@{����?L��@�  C�
=@{���  @\(�A陚C�Ф                                    Bx���  
�          @�@vff����?z�HA�HC��@vff����@b�\A�(�C�4{                                    Bx��t  "          @�=q@p  ��p�>�Q�@C33C���@p  ��=q@@  A�
=C���                                    Bx��  
�          @��H@p����{����  C��)@p������@*�HA��C�ff                                    Bx���  "          @�  @�\)����?G�@�z�C���@�\)��  @S33A�33C��R                                    Bx��f  
�          @�\)@������?\(�@�33C�u�@�����R@XQ�A��C�z�                                    Bx�  "          @�z�@�ff��>�ff@p  C��R@�ff��=q@>{A���C�H�                                    Bx��  �          @��
@|(����
        C��H@|(���ff@*�HA�
=C�AH                                    Bx�"X  
Z          @�33@�\)��z�=��
?&ffC�\@�\)���R@(Q�A�z�C���                                    Bx�0�  
�          @�ff@�=q��33?(��@�
=C�H@�=q���@P  Aޏ\C��
                                    Bx�?�  "          @���@��R����<�>��C�  @��R���
@"�\A�Q�C�                                      Bx�NJ  �          @�  @�p���  �p����C�u�@�p����
?�Q�AJ�RC��R                                    Bx�\�  T          @��@��H�����
��z�C�9�@��H��=q>��@�C���                                    Bx�k�  �          @Ӆ@�G���=q�z����\C�0�@�G���G�>u@�C���                                    Bx�z<  �          @ҏ\@�\)��ff�ٙ��pz�C�33@�\)��  ?�\@�  C�7
                                    Bx���  �          @��@�����33�����
=C�>�@������\?��HA)C�N                                    Bx���  
�          @��@�
=���R�n{��HC��q@�
=���H?�
=AJ{C�!H                                    Bx��.  �          @�(�@�ff���Ϳ   ���C���@�ff��=q?��Az=qC���                                    Bx���  �          @�33@�Q�����.{��p�C��f@�Q����?�
=AlQ�C���                                    Bx��z  �          @��
@�Q���=q�Ǯ�W�C��@�Q���p�?�(�A��RC��                                    Bx��   �          @��@�ff���\��33�EC�Z�@�ff����@ ��A�{C��=                                    Bx���  T          @�  @�����
��(���=qC�q@����  >��H@���C���                                    Bx��l  T          @�Q�@�����׿�Q���\)C�8R@�����
?��@��\C�<)                                    Bx��  T          @�  @�{��G������
C�'�@�{���\>B�\?�\)C���                                    Bx��  
�          @��@�  �����޸R�mp�C�)@�  ���\?(��@�33C�B�                                    Bx�^  �          @�z�@�  ��G��ff��C�(�@�  ���>�33@8��C��                                    Bx�*  �          @�\)@�33��zῷ
=�Dz�C�t{@�33��=q?@  @��
C�޸                                    Bx�8�  
�          @�  @�33���
�У��`  C���@�33����?\)@�
=C��H                                    Bx�GP  
�          @��@��������z�� z�C��@�����?p��A�C���                                    Bx�U�  �          @��
@��H��녿�\)��
C��)@��H���H?z�HA��C��q                                    Bx�d�  
Z          @��
@����c�
�G��ٙ�C��3@����`  ?��A��C���                                    Bx�sB  �          @��H@����`  ����\)C��@����U?�Q�A'
=C�~�                                    Bx���  �          @Ӆ@����I�������]p�C�@����<(�?�
=A$  C���                                    Bx���  	�          @�(�@��\�I��>#�
?�{C��\@��\�+�?�Ah��C��=                                    Bx��4  T          @Ӆ@����L��=�Q�?G�C�xR@����0��?У�Aep�C�8R                                    Bx���  �          @��@�ff�W��\�P��C��
@�ff�HQ�?��A5C��H                                    Bx���  
�          @Ӆ@���i��������C�33@���Z=q?�\)A?33C��                                    Bx��&  
Z          @أ�@���q녿��
�R�\C�h�@����G�?   @�  C�y�                                    Bx���  T          @��
@���p  �Q����
C�5�@������  ��C��                                    Bx��r  
(          @���@����r�\�%���  C���@�����=q����Y��C�8R                                    Bx��  
�          @�@�G��x���/\)��ffC�+�@�G���\)�   ���C�ff                                    Bx��  
�          @�@��R�XQ��E���
=C���@��R��ff����
{C��H                                    Bx�d  �          @ٙ�@����aG��<����\)C�|)@�����Q�Y����ffC��                                    Bx�#
  
�          @���@��R�e�����C��)@��R��녾�Q��EC�0�                                    Bx�1�  
�          @��@����G��U����C�&f@�����33�����;�C��                                     Bx�@V  
�          @׮@�
=�H���P  ��C��3@�
=���\����1�C�q�                                    Bx�N�  �          @ڏ\@�Q��9���|���G�C�N@�Q����R�G����C�h�                                    Bx�]�  T          @��@���*�H��ff�
=C��@����(��ff��p�C�`                                     Bx�lH  "          @�z�@�G��#�
����33C�K�@�G��������z�C�s3                                    Bx�z�  �          @�p�@����.�R��=q�!��C���@�����  ��H��z�C��H                                    Bx���  
Z          @ָR@��H�1���G��*33C���@��H�����%��p�C���                                    Bx��:  �          @�(�@��\�-p���\)�)��C�1�@��\����$z���(�C��                                    Bx���  
�          @��H@`  �#�
��ff�C33C��f@`  ����C33��
=C�o\                                    Bx���  T          @Ӆ@p  �)����
=�7=qC�^�@p  ��(��3�
���
C�~�                                    Bx��,  T          @��H@��R�{��p��(C��=@��R���\�(Q���z�C�R                                    Bx���  
�          @�Q�@�\)�!G�����"�C��3@�\)��G��(����HC�L�                                    Bx��x  
�          @�G�@����(Q������=qC�8R@������H�z���ffC�S3                                    Bx��  T          @�\)@�
=�'
=�e��Q�C���@�
=��G���=q��G�C���                                    Bx���  
�          @�
=@�{�/\)�p  ���C�.@�{��\)��z���=qC�1�                                    Bx�j  "          @���?�Q�?�Q������B3��?�Q쿹�����R
=C���                                    Bx�  
�          @�(�?k�?޸R���R�\Bx�?k���Q���G�p�C�@                                     Bx�*�  
�          @��@C33������H�PC�
@C33�����G�����C���                                    Bx�9\  "          @˅@p������p��r��C�H@p�����tz����C�
                                    Bx�H  
�          @θR@�׿�������C��@���xQ����H�6z�C��                                    Bx�V�  "          @�?��?\)���
�RA�?���"�\��33�{��C��f                                    Bx�eN  "          @Ϯ?�?�����\).B��?��
=q��{�C���                                    Bx�s�  
�          @���?��?����(��HB3��?�����H���H��C��                                    Bx���  �          @�{?\(�@AG����
�s
=B�8R?\(��#�
��(�¤\)C��q                                    Bx��@  �          @��H?fff@7
=��33�{\)B��?fff��
=��  ¢�C�xR                                    Bx���  �          @�G�?�@%���=q�|�RBt?��(���ʏ\G�C��
                                    Bx���  �          @�{?���@]p���Q��[�B�\)?���>\��G�ffA�{                                    Bx��2  "          @�
=?��@I����G��k�
B��3?������(� �C�S3                                    Bx���  
�          @У�?E�@>{����w�B��)?E���\)�θR¥z�C��                                    Bx��~  �          @�녾k�@&ff��Q��B�G��k��:�H��Q�¦z�C|J=                                    Bx��$  
�          @љ��@  @ ����Q��HB�{�@  �Q���
= �\Cc�H                                    Bx���  T          @��H�5?�����(��B�aH�5�����\)�qCyz�                                    Bx�	p  T          @��>k�?��������B��f>k������(��)C��3                                    Bx�	  
�          @�G��   @(����{L�B�=q�   �.{��
=¥8RCi��                                    Bx�	#�  T          @�ff��  @����u�B�녿�  ������\)�3Cd��                                    Bx�	2b  
�          @����
=?����G�G�B��3��
=�˅���HQ�CiaH                                    Bx�	A  "          @�\)���H@p����H�RB�uÿ��H�k���  G�CT+�                                    Bx�	O�  �          @�{��@!G�����B��H���W
=��
=p�CR��                                    Bx�	^T  
�          @�ff���H@+���ff�{33B�33���H�(���ϮB�CL�                                    Bx�	l�  
�          @�{��  @
=��  �B���  �xQ���(�B�CP�                                    Bx�	{�  
�          @�����H?�p�����aHB������H��\)��=q�HC_+�                                    Bx�	�F  T          @�p����H@{�����B�.���H�c�
�θRW
CSJ=                                    Bx�	��  �          @�33?���<��
��
=��?p��?����>�R��p��j{C���                                    Bx�	��  T          @��Ϳ�
=?��
�����B�p���
=������Q�B�Ci
=                                    Bx�	�8  �          @�G��1G�@?\)����M�RCٚ�1G�<��
��p���C3��                                    Bx�	��  T          @����'
=@
=��33�l=qC��'
=�z�H����z�CH��                                    Bx�	ӄ  �          @�(�� ��@����
=�m�C	��� �׿k��\� CH
=                                    Bx�	�*  
�          @�\)�#33@
=��  �kz�C	!H�#33�Y������fCFxR                                    Bx�	��  T          @Ӆ�2�\@(�����
�YC�\�2�\��
=��
=p�C<��                                    Bx�	�v  "          @��H� ��@{��=q�f��CaH� �׿+������CB�f                                    Bx�
  "          @�(��+�@������c��C	Ǯ�+��333������CB�=                                    Bx�
�  T          @��H�8��@p���  �b{C�H�8�ÿ\(�����{{CD�f                                    Bx�
+h  
�          @љ��2�\?�Q����H�j�C.�2�\��������zG�CJT{                                    Bx�
:  T          @ə��B�\?�G�����B���B�\���R���Cx�R                                    Bx�
H�  "          @���@O\)�O\)����M=qC��@O\)�333�L(����C���                                    Bx�
WZ  T          @�Q�@Dz�?h������W�
A�(�@DzῸQ���Q��M�C�w
                                    Bx�
f   T          @��
@g�?����p���3A�  @g��z�H�r�\�5Q�C�q�                                    Bx�
t�  	.          @�  @��H?k��L(���HA>�\@��H�Q��Mp��33C���                                    Bx�
�L  
�          @�p�@��@�
��=q�m(�B,33@�ͿJ=q��p�G�C�"�                                    Bx�
��  
�          @��H?�33?�����#�B  ?�33������(��C�Z�                                    Bx�
��  
�          @��?�z�?n{���G�A���?�z��ff��p��w�RC�G�                                    Bx�
�>  
�          @�z�@G�?�=q��33�\A���@G���33��ff�x�C�j=                                    Bx�
��  �          @�z�?��?ٙ���  =qBj{?��������33�{C��                                    Bx�
̊  T          @��>#�
@p���{�B�>#�
�8Q���33¥  C�>�                                    Bx�
�0  �          @���?n{?�Q����
�=Bd\)?n{��p�����)C�q                                    Bx�
��  �          @�=q?�{?�{��ff\B3�
?�{�˅��z��C�P�                                    Bx�
�|  T          @��\?�ff@�
��=q  B{�R?�ff�k���(�{C�aH                                    Bx�"  �          @�Q�?aG�?�����
��B�?aG��������\�3C�O\                                    Bx��  �          @�=q?xQ�@�����R�~p�B�\?xQ�z���ff��C���                                    Bx�$n  "          @��?\)@Q���{�{B�� ?\)�
=��£C��R                                    Bx�3  
�          @�>8Q�@'������y�\B��>8Q쾔z����¬��C�{                                    Bx�A�  �          @�{>�\)@333���R�q�B��\>�\)������{­��C��                                    Bx�P`  T          @���>��@*=q�����sz�B�
=>��������­�)C�
=                                    Bx�_  	�          @���?�@33��ǮB�z�?������¤z�C��                                    Bx�m�  
�          @���?L��?�Q���ffW
Bt�?L�Ϳ�33����p�C�`                                     Bx�|R  
(          @���>�33@  ��(��G�B�aH>�33��
=��(�§�)C���                                    Bx���  �          @�Q�?�Q�?��\�r�\�|��B%Q�?�Q�W
=�z=q��C��f                                    Bx���  
�          @���>�?�{����B��>�������¡�\C���                                    Bx��D  
�          @�33��(�@%��=q�X�
B����(�<���G�B�C2�H                                    Bx���  
�          @�
=�33@=p���(��C��B���33>�G������C'�                                    Bx�Ő  �          @��R��G�@>�R�����RffB�{��G�>����p���C+��                                    Bx��6  �          @��R���@'�����]��B�{��논���ff��C533                                    Bx���  �          @�=q��ff@1�����\=qB�B���ff=�\)���H��C1xR                                    Bx��  	�          @�녿�p�@:=q��  �X�B��
��p�>W
=����
C+�                                    Bx� (  "          @�녿�  @AG�����W�
B�{��  >�������(�C&W
                                    Bx��  T          @�p�����@2�\��ff�]
=B�{����>���  �C.u�                                    Bx�t  
�          @�\)��=q@2�\�����^�B��f��=q=�Q�����HC0#�                                    Bx�,  "          @����  @+����a  B����  =#�
��C2�                                    Bx�:�  �          @��\�Y��@E���z��c�\B��ͿY��>������£{C*E                                    Bx�If  T          @����L��@�
���R��B��3�L�;�
=��\)ª�3C�T{                                    Bx�X  "          @�  >�@ff������B��>��E�����£�\C�˅                                    Bx�f�  �          @��H>��@
=q����ǮB�� >�녿h����� Q�C�!H                                    Bx�uX  �          @��=��
@,(���{�p�RB��\=��
��Q����±u�C��                                    Bx���  �          @�{��@=q����~��B�G�����
=���«(�C�'�                                    Bx���  �          @�z�.{@*=q��ff�x�RBг3�.{�\���\¤��CP��                                    Bx��J  �          @����\@?\)��Q��kQ�BǊ=��\    ���\©�C3�q                                    Bx���  �          @�{�u@C33��ff�_(�B��u>W
=��33 
=C'��                                    Bx���  
�          @��5@G������]�B͔{�5>������H¤\)C!H                                    Bx��<  �          @�\)�5@,���O\)�EG�B�Q�5?(����p���C	�                                    Bx���  
�          @n�R?�?����I���hBT�
?���=q�`��33C���                                    Bx��  �          @�  ?�=q?�=q�g��np�A�p�?�=q�k��j=q�s=qC��3                                    Bx��.  �          @`  ?���?�\�B�\�t��A�
=?��Ϳ�\)�8Q��_��C�~�                                    Bx��  �          @�\)?�ff�k��j=q�~�\C�L�?�ff�z��AG��>��C�s3                                    Bx�z  
Z          @��\@R�\�\)�3�
��HC��@R�\�W
=���\�q�C�4{                                    Bx�%   �          @���@U�:=q�5��(�C�w
@U�|�Ϳp���$  C�q                                    Bx�3�  
�          @��@^�R�g
=��
=�T��C�H@^�R�l��?Tz�A��C��                                     Bx�Bl  �          @�@b�\�'
=�,����(�C���@b�\�g��z�H�.�HC�.                                    Bx�Q  
Z          @��
@Dz��XQ�����{C�q@Dz���=q�aG���RC��H                                    Bx�_�  
�          @�33@J�H�L(�����z�C�n@J�H�|�;�������C�`                                     Bx�n^  "          @��
@e��0��������C�4{@e��e���R��p�C��H                                    Bx�}  
Z          @�Q�@{��+���R��33C�� @{��\(��
=q��C�e                                    Bx���  "          @���@�=q����	������C�~�@�=q�?\)�5���RC��f                                    Bx��P  T          @�G�@��׿�Q쿪=q�l  C�AH@����Q쾏\)�A�C��{                                    Bx���  "          @�z�@�ff�&ff�aG��33C���@�ff�?�33AEG�C��f                                    Bx���  "          @�ff@��\�P  ?   @���C���@��\�!�@ffA�C��
                                    Bx��B  �          @��H@�  �E?c�
A=qC���@�  �(�@ffAѮC�W
                                    Bx���  "          @�ff@�p��W
=?
=@�Q�C���@�p��%�@�RA�C��                                    Bx��  T          @�z�@�(��A녿!G���{C���@�(��:�H?��\A/�C�%                                    Bx��4  �          @�Q�@tz��g�����ffC�L�@tz��W�?�z�Av�HC�K�                                    Bx� �  "          @��@��\�G��33����C�P�@��\�n�R�������C��\                                    Bx��  
�          @�33@�33�`��>�Q�@n�RC���@�33�5�@�A�C�|)                                    Bx�&  
�          @��
@�  �QG�?�@��HC���@�  �"�\@Q�A�
=C�H�                                    Bx�,�  
�          @�=q@C33���H���H�L��C�T{@C33�l���5��\)C��)                                    Bx�;r  
�          @���@E������D(�C�@E��r�\�&ff��(�C��=                                    Bx�J  T          @���@5�������U  C��q@5�xQ��<(����C�!H                                    Bx�X�  T          @�\)@8Q��G����\�W��C�y�@8Q���
=�Fff��z�C�*=                                    Bx�gd  �          @�G�@#�
��{����j=qC��R@#�
�g
=�W��{C���                                    Bx�v
  �          @�33@P�׿�  ��{�QC��H@P���H���R�\�(�C��                                    Bx���  
�          @���@R�\��G������I�C���@R�\�QG��@����C���                                    Bx��V  T          @�{@N{��33�����W(�C�G�@N{�j�H�X���Q�C���                                    Bx���  �          @�{@c33�xQ���z��NQ�C�XR@c33�N{�^�R��C��                                    Bx���  �          @��@W
=�����(��R�C��)@W
=�Tz��Z=q�Q�C��=                                    Bx��H  
�          @��H@Dz��z�����T�RC��=@Dz��tz��H����C�aH                                    Bx���  "          @��@J�H�J=q�����aQ�C�@J�H�L���s33� ��C�`                                     Bx�ܔ  "          @�ff@J=q���H���R�_p�C���@J=q�5�r�\�&�HC���                                    Bx��:  T          @��@[��Tz���p��S�C�,�@[��G��e����C��q                                    Bx���  
�          @��@L�Ϳz������_�C��@L���=p��s33�$z�C���                                    Bx��  �          @��@\��>��������U\)@�  @\���Q���z��6p�C�                                      Bx�,  �          @��R@7�=L����p��o
=?�  @7��\)����@C��                                    Bx�%�  �          @�@0  >���
=�t(�@,(�@0  ������\�F�C�'�                                    Bx�4x  �          @�  @1�>�������s�@@��@1��p���(��F�HC�C�                                    Bx�C  "          @��@0�׼��
��33�v=qC�Ǯ@0���(������C{C�!H                                    Bx�Q�  T          @�
=@3�
��p����R�p�HC�Ff@3�
�7�����5G�C�:�                                    Bx�`j  "          @�\)@>�R�\)��(��j(�C���@>�R�(Q�����7��C�J=                                    Bx�o  �          @�p�@1G�>�z���{�r=q@��@1G���\��(��J��C�9�                                    Bx�}�  
�          @��@7�=������lff@�@7������AG�C�AH                                    Bx��\  T          @�G�@%>�Q����
�w�@��@%�(�����P�C�ٚ                                    Bx��  
�          @�{@.{>�
=��ff�nz�A�@.{��
����LC�s3                                    Bx���  T          @�\)?�p�?�G�������B{?�p����
��{�C�G�                                    Bx��N  "          @�(��)��@33�����V
=CE�)������{�t��C@.                                    Bx���  
�          @��\�  @33����cz�C	��  �#�
����L�CC�                                    Bx�՚  �          @��
�#33@z������XG�C�{�#33�����ff�y  C@=q                                    Bx��@  
�          @���>\)?��H��
=��B�ff>\)��
=��\)8RC��3                                    Bx���  �          @�ff?���?B�\���R8RA���?����G�����|�HC�0�                                    Bx��  T          @�?���?���  #�A��H?�������\�z�HC��                                    Bx�2  "          @����
=?\)���¤�RB�  ��
=�p���G��=C��
                                    Bx��  �          @�\)?�\?\)����k�A���?�\�
�H��ff�l�HC���                                    Bx�-~  �          @��?�
=?B�\�����B(�?�
=�ff��\)aHC���                                    Bx�<$  T          @�z�?(�?E���G��BN��?(��(�����aHC��                                    Bx�J�  �          @��?�?Y����(�k�Ba�R?���������C��=                                    Bx�Yp  T          @�33@?\)>k���Q��lff@�33@?\)�Q���p��Dp�C��)                                    Bx�h  
�          @��H?�{?�Q�����fB  ?�{��p����R(�C���                                    Bx�v�  T          @��׽�G�?5��¦B�B�  ��G��(���ff��C���                                    Bx��b  �          @���?\)?���£�B5�\?\)����p�p�C���                                    Bx��  T          @�z�?��R?s33��33u�B?��R��
=���\�qC�˅                                    Bx���  �          @���?���?�=q���
� A�R?��Ϳٙ���ff�|=qC��)                                    Bx��T  "          @��@   ?��R��33�p\)A�p�@   ��
=�����l=qC�#�                                    Bx���  �          @�?��?��\���z�B��?�׿�Q����(�C�K�                                    Bx�Π  �          @�ff@��?�=q�����{  A�(�@�ÿ��H��G��q=qC��\                                    Bx��F  T          @�
=@.{?����33�f��A�G�@.{��(���\)�]��C��f                                    Bx���  �          @�=q@7
=?�z���(��az�A���@7
=������=q�\�C��                                    Bx���  T          @��@(Q�?W
=�����m�A���@(Q����p��[Q�C��{                                    Bx�	8  T          @��@?=p����H�w�\A�(�@��(���=q�`Q�C��\                                    Bx��  �          @�{@333?���Q��Z{A��
@333��
=��Q��Y��C��3                                    Bx�&�  �          @�G�@w
=?�ff�c33�33A�Q�@w
=��=q�~{�7  C��R                                    Bx�5*  �          @��H@�Q�?�ff�H����RA��@�Q콏\)�g
=�!\)C���                                    Bx�C�  
�          @�(�@�ff@   �0����ffA�G�@�ff>����Y���33@s33                                    Bx�Rv  T          @�(�@�
=@
�H��ff�W�
A�ff@�
=?�Q���R��ffAX                                      Bx�a  "          @��@��H@	���s33��A���@��H?�{��z����An�H                                    Bx�o�  �          @���@��?�Q�!G�����A���@��?���������Aip�                                    Bx�~h  T          @��
@�
=?�z�>�33@q�AH��@�
=?�
=��z��G�AK�                                    Bx��  
(          @���@�33?�녽u�(�Ah��@�33?�z�B�\��AD(�                                    Bx���  
�          @�ff@�Q�?����z��B�\A:�H@�Q�?J=q�L���	G�AQ�                                    Bx��Z  �          @��@�ff?�=�?��HA�Q�@�ff?��H�aG���HA��                                    Bx��   
�          @���@�G�@�
=L��?��A��H@�G�@ �׿���=G�A��                                    Bx�Ǧ  
�          @�ff@�@\)��ff��
=A�p�@�?ٙ����
����A�                                    Bx��L  
�          @�{@�\)?�(��aG���A�ff@�\)?��
�������A:�\                                    Bx���  �          @�  @��H?��Ϳ������A�(�@��H?�����H�x��Ac33                                    Bx��  T          @���@�33@'���\��G�A�p�@�33?��R���
��(�A��H                                    Bx�>  
�          @���@�p�@(�����S
=A��@�p�?�Q��ff��p�A�ff                                    Bx��  "          @��R@�@   � ����A��\@�?0���0  ��p�@�{                                    Bx��  
Z          @��R@�=q?����C�
��RA�  @�=q�\�U��
=C���                                    Bx�.0  	�          @��R@�  ?n{���H����A (�@�  �W
=�
=q���RC��                                     Bx�<�  	�          @�@�{>aG��
�H��p�@�@�{�k���p����\C��R                                    Bx�K|  T          @�@�    �G���
=C�  @���z������33C��3                                    Bx�Z"  S          @��\@�(����E����C���@�(������(Q���{C�t{                                    Bx�h�  �          @�{@�Q�>��
�Tz��\)@���@�Q쿵�AG����C���                                    Bx�wn  �          @�p�@�ff>��
�Dz��
�@�p�@�ff����3�
���C��                                    Bx��  �          @�Q�@�p�?#�
�L(��(�AQ�@�p���=q�Dz���C�(�                                    Bx���  �          @��R@�33���Ϳ�����p�C�@�33��þ������C�j=                                    Bx��`  �          @���@��R��ff�����(�C�  @��R�p���
=�<��C��                                     Bx��  �          @�G�@��
���\����£�C��@��
��
��z��`��C��                                     Bx���  �          @�
=@�ff�����\)���C��@�ff���p���z�C�)                                    Bx��R  �          @��R@�p�����������HC�#�@�p��G���ff�R�RC�Y�                                    Bx���  �          @�{@��G��{��\)C��R@���ff��G��v{C�k�                                    Bx��  �          @��H@��ÿ(����  C���@��ÿ��H�ٙ���{C��)                                    Bx��D  �          @�=q@��׿��
�"�\��C�|)@�����������Q�C��=                                    Bx�	�  �          @��\@�녿��#33�ٙ�C�/\@���33������\)C�9�                                    Bx��  
�          @��@��Ϳ@  � ����33C���@��Ϳ�
=���
���C�C�                                    Bx�'6  �          @�=q@��H����,����{C�:�@��H���
����  C���                                    Bx�5�  �          @�Q�@�녾B�\�*=q��\)C�ٚ@�녿\�(����C�=q                                    Bx�D�  �          @�ff@�����
�*=q��ffC��@�����  �îC���                                    Bx�S(  �          @�{@�  �����&ff���HC���@�  ����\��  C�Q�                                    Bx�a�  �          @�
=@�{�Ǯ�1����RC���@�{��  ������C��H                                    Bx�pt  �          @�\)@��R��33�1G�����C�� @��R���H�p���=qC�
=                                    Bx�  �          @��R@�=q�k��"�\��Q�C���@�=q���R������C�n                                    Bx���  �          @��@�ff��\)�5���Q�C���@�ff��  ����υC�'�                                    Bx��f  �          @�G�@�(�����A���RC��@�(����!����C�R                                    Bx��  �          @�z�@���
=q�-p���{C��)@������
���
C���                                    Bx���  �          @�  @�Q쾨���4z���\C�q@�Q���H�����33C��\                                    Bx��X  �          @�\)@��?:�H�   ��z�A\)@�녿#�
�!G��ҸRC�j=                                    Bx���  �          @�G�@�\)?Y���O\)�  A#33@�\)�h���N�R�G�C���                                    Bx��  �          @�Q�@J=q@%����\�3�B{@J=q>#�
��=q�c  @>{                                    Bx��J  �          @�
=@@  @#33��(��9z�B!p�@@  =�G���33�i
=@
=                                    Bx��  �          @���@3�
@>�R����4G�B:��@3�
>��H�����r{A\)                                    Bx��  �          @��@P��@,�������-��B��@P��>������\�_�@�G�                                    Bx� <  �          @�33@4z�@X���vff�$��BI(�@4z�?z�H��G��m�A��                                    Bx�.�  �          @�G�@E@L���k��\)B8�@E?h�������`p�A���                                    Bx�=�  T          @��H@L(�@J�H�l����
B3ff@L(�?^�R��G��]�RAt                                      Bx�L.  �          @��H@`  @L���W���\B)p�@`  ?�������L  A��\                                    Bx�Z�  �          @�33@QG�@K��j�H�G�B0@QG�?c�
�����Zp�At��                                    Bx�iz  �          @��\@P��@G��hQ��{B.�H@P��?^�R��ff�YQ�An=q                                    Bx�x   �          @�33@1�@j�H�e��ffBS��@1�?�\)��{�g�A�                                      Bx���  T          @���@Q�@����N�R���Bs�H@Q�@z���ff�f{B$=q                                    Bx��l  �          @�{@-p�@�33�W
=�	��BbG�@-p�?�����
=�bffB	\)                                    Bx��  �          @�33@)��@���=p�����Bd�\@)��@�
���
�X{BQ�                                    Bx���  �          @���@��>�p��=q��  @�{@�녿c�
�G�����C�Ǯ                                    Bx��^  �          @��R@����{�����C���@���{��{�H  C���                                    Bx��  �          @�ff@��׿ٙ���p���{C��{@�����&ff��C�P�                                    Bx�ު  �          @���@�33?}p�?�G�AZ�\A/�
@�33?��R?��@�z�A�Q�                                    Bx��P  �          @�
=@��R<��
?�=qA���>W
=@��R?O\)?�{Ag
=Ap�                                    Bx���  �          @�Q�@�  ���þB�\��C���@�  ����?!G�@�ffC��H                                    Bx�
�  �          @���@�(��Fff��z��:=qC���@�(��5�?�ffAO\)C��{                                    Bx�B  �          @��@����W������
C�y�@����;�?�z�A�p�C�K�                                    Bx�'�  �          @�z�@�=q�~�R>�G�@���C��3@�=q�N{@�A��C���                                    Bx�6�  �          @��@\)����?�\@���C���@\)�\(�@%�A��HC���                                    Bx�E4  �          @�G�@g
=���\?\Ab=qC�ff@g
=�Z�H@n�RBz�C�G�                                    Bx�S�  �          @��H@(�����
@\)A��\C���@(���p  @�  B8  C��{                                    Bx�b�  �          @�  @%���\@G�A�G�C�u�@%�u�@�G�B1�HC��                                    Bx�q&  �          @�(�@"�\��(�?�
=Ak33C��@"�\��  @�z�B&=qC�j=                                    Bx��  �          @��@H����
=?�\)A�{C��@H���s�
@�33B%�\C�                                    Bx��r  �          @љ�@G
=���R?�(�Aup�C�Ф@G
=�w�@�
=B!ffC�]q                                    Bx��  �          @�Q�@��R��{?��AG�
C��=@��R�J=q@Z�HB \)C�XR                                    Bx���  �          @У�@�{��(�?��A�C�y�@�{�Q�@EA��C��                                    Bx��d  �          @�G�@�Q�����?��
A4z�C��R@�Q��`��@_\)B33C�@                                     Bx��
  �          @�(�@�������?��A(�C���@����n{@W
=A��RC���                                    Bx�װ  �          @У�@^�R��  ?�{Ae��C��=@^�R�p  @�Q�B=qC�p�                                    Bx��V  �          @У�@]p���=q?�33AFffC���@]p��{�@w
=B�\C��3                                    Bx���  �          @�G�@s33����?�{AC�/\@s33�z�H@a�Bz�C��                                    Bx��  �          @�G�@`�����R?:�H@��C�` @`�����
@UA�Q�C�]q                                    Bx�H  �          @�(�@Tz���33?���A'�C�Y�@Tz���Q�@tz�B{C��R                                    Bx� �  �          @�{@Tz����?�\)A (�C�˅@Tz���z�@i��B
(�C�b�                                    Bx�/�  T          @�{@W
=��p�?c�
@�p�C��f@W
=����@]p�B�HC�)                                    Bx�>:  �          @�Q�@l����33?ǮA^�HC���@l���i��@xQ�B�C���                                    Bx�L�  �          @�z�@w��s�
@4z�A�\)C��R@w���@��HB4ffC��3                                    Bx�[�  �          @�p�@i����p�?�p�A�ffC��@i���L��@tz�B��C�aH                                    Bx�j,  �          @��@p  ���?��
A��C�(�@p  �O\)@x��B�C��3                                    Bx�x�  �          @�@mp���33?�p�A�ffC�t{@mp��H��@qG�B=qC��q                                    Bx��x  �          @љ�@�z���?��
A5�C��
@�z��\(�@[�A���C���                                    Bx��  �          @��@�����33?��\A(�C�t{@����`  @J=qA�=qC�]q                                    Bx���  �          @�(�@������?^�R@��C��3@����s33@I��A���C�T{                                    Bx��j  �          @�p�@�ff���R?��
AT��C��\@�ff�U@j=qB(�C��=                                    Bx��  �          @˅@\)���
?z�@���C���@\)�|��@9��Aڣ�C��                                    Bx�ж  �          @ȣ�@�����\)?8Q�@�(�C�:�@����p��@<��A�C��                                     Bx��\  �          @�  @�����z�?\)@�p�C���@����p��@1G�A�z�C��f                                    Bx��  �          @�\)@�(���=�\)?#�
C�T{@�(��tz�@\)A�ffC�n                                    Bx���  �          @�z�@�����\)�����
C�\@����}p�?�=qAn�\C���                                    Bx�N  �          @ƸR@z=q��33=���?n{C�k�@z=q��@{A��RC��=                                    Bx��  �          @�p�@i����
=>L��?��C�"�@i�����@'�Aȏ\C�`                                     Bx�(�  T          @�ff@G���33>�33@Q�C�  @G����@;�A�C�e                                    Bx�7@  �          @Å@�
=���;�=q�%�C��\@�
=�p  ?��A��\C��q                                    Bx�E�  �          @��
@��\��G�>��?��C���@��\�j�H@�RA�\)C�޸                                    Bx�T�  �          @\@������>#�
?��RC��f@���i��@{A��C��                                    Bx�c2  T          @���@��R���H����{C��@��R�u@G�A��C���                                    Bx�q�  �          @���@��H�a녿(���=qC��3@��H�XQ�?�Q�A6=qC��=                                    Bx��~  �          @�33@����{��33�9C��@����.{>L��@�C��f                                    Bx��$  �          @�p�@����u�=�Q�?k�C��@����S33?���A�33C��                                     Bx���  �          @�G�@����n{�\)���C��f@����Tz�?�Q�A���C���                                    Bx��p  �          @���@����{����
�=p�C��@����^{?�A�C��                                     Bx��  T          @�G�@�z��n�R>k�@�\C��\@�z��I��@G�A��C�*=                                    Bx�ɼ  �          @��R@�(��n{��  �=qC��R@�(��XQ�?˅Ax(�C��3                                    Bx��b  �          @��@�=q�w
=�#�
���
C��@�=q�XQ�?�\)A��C���                                    Bx��  �          @��H@���}p�=#�
>�p�C���@���\(�?���A�{C��                                     Bx���  �          @�ff@�
=��Q�<��
>k�C�Ф@�
=�`  ?�(�A�  C���                                    Bx�T  �          @��H@�ff��(�>Ǯ@k�C���@�ff�[�@A��C�5�                                    Bx��  
�          @��
@������>Ǯ@l��C���@���dz�@�A�G�C�/\                                    Bx�!�  T          @�@��H��=q?333@ϮC��=@��H�j=q@4z�AمC�{                                    Bx�0F  �          @�p�@��\���
>�z�@)��C��)@��\�y��@�RA��C�!H                                    Bx�>�  �          @�p�@|(����R>��R@7�C��\@|(��~�R@"�\AÅC�XR                                    Bx�M�  �          @�33@l(�����?Q�@�\)C���@l(��r�\@AG�A��
C�!H                                    Bx�\8  �          @�z�@R�\���?�G�AQ�C���@R�\�|��@U�B��C��                                    Bx�j�  �          @\@c33��\)?���AQ�C�k�@c33�_\)@_\)B(�C��                                     Bx�y�  T          @��H@Q���{?��AK�C�˅@Q��l(�@c33B��C���                                    Bx��*  �          @�{@s�
��=q?E�@���C�(�@s�
�w
=@?\)A�33C�Q�                                    Bx���  �          @�
=@�(���=q?E�@���C�f@�(��h��@7�Aܣ�C�H�                                    Bx��v  �          @�p�@�(����=�G�?���C�
@�(��|(�@�\A���C�(�                                    Bx��  �          @�ff@j=q���\?uA��C�Q�@j=q�c�
@B�\A�  C��                                    Bx���  �          @�33@i����ff?uA(�C���@i���\��@>�RA�\)C�W
                                    Bx��h  �          @���@g����H�8Q��  C�#�@g����@�\A�33C���                                    Bx��  �          @�ff@dz���ff>�  @�HC���@dz���Q�@{A�p�C��
                                    Bx��  �          @��@k����>�p�@dz�C�(�@k��z�H@#33A�
=C��q                                    Bx��Z  �          @�
=@L������?�p�A>{C��3@L���n{@Z�HB�C�Z�                                    Bx�   �          @��@��
�s33�(���ָRC���@��
�i��?�p�AI��C�9�                                    Bx��  �          @�z�@|(��z�H�Y���
�RC��{@|(��vff?�\)A8  C���                                    Bx�)L  �          @�=q@S�
�c33�G���=qC���@S�
��ff���R�QG�C�&f                                    Bx�7�  �          @�
=@fff�o\)��  ���C��R@fff��  >�@��\C���                                    Bx�F�  �          @�Q�@O\)��
=�����h  C���@O\)���?J=qA��C�O\                                    Bx�U>  �          @�z�@?\)��  ��ff��ffC�Ǯ@?\)��ff?E�@�33C�7
                                    Bx�c�  �          @�z�@U�z�H�\)����C�8R@U��Q��G���{C�C�                                    Bx�r�  �          @�  @����y�����R�G�C��@�������?B�\@�\C��                                     Bx��0  !          @�ff@u��33�k��p�C��R@u����?�\)A6=qC��                                    Bx���  �          @�  @��
��G���ff���C��@��
�\)?У�A~=qC��
                                    Bx��|  �          @�(�@����|(������*�HC��3@�����  ?\(�A ��C�S3                                    Bx��"  �          @��R@�
=�y�������(z�C��H@�
=�|��?fffA��C�y�                                    Bx���  �          @���@��x�ý����HC�]q@��_\)?�(�A�  C��                                    Bx��n  �          @��@����]p�>�Q�@`  C�@����8Q�?���A�(�C�n                                    Bx��  �          @��@�p��H��?�z�AaG�C�\@�p��ff@.{A��
C��                                     Bx��  �          @��\@��H�?\)?�Q�A�ffC�u�@��H��(�@G�B��C��3                                    Bx��`  �          @�33@�p��AG�?�=qA��
C���@�p���ff@A�A�C�p�                                    Bx�  �          @�33@s33�j�H?s33A#\)C�@s33�2�\@#�
A��C���                                    Bx��  �          @���@I�����������C��=@I�����R@\)A��C�7
                                    Bx�"R  �          @���@U���논��
�#�
C��@U���ff@{A�C���                                    Bx�0�  �          @��
@`  ����>�33@H��C��f@`  ��Q�@1�Aϙ�C��                                    Bx�?�  �          @��@C33��  >��@n{C���@C33��p�@;�AݮC��{                                    Bx�ND  �          @�z�@H����Q�?&ff@��
C��R@H����=q@I��A�RC�AH                                    Bx�\�  �          @�@A����\=���?z�HC��3@A���p�@%�A���C���                                    Bx�k�  �          @��@����k������QG�C�\@����Z=q?�Aep�C�!H                                    Bx�z6  �          @��R@�ff��Q�(����\)C�ٚ@�ff��{������C�H                                    Bx���  T          @��
@�p���  �J=q� Q�C�B�@�p��ٙ�    �#�
C�N                                    Bx���  �          @��@���^�R��33��  C���@�����
����(  C�4{                                    Bx��(  �          @���@�
=�
=��=q�.=qC���@�
=��<��
>B�\C�Ǯ                                    Bx���  �          @���@�p��������1G�C��3@�p��(�=#�
>�Q�C�^�                                    Bx��t  �          @��
@�33��G���=q�[
=C���@�33��G��5��C��                                    Bx��  �          @�p�@����p���  �v=qC��@��� �׿#�
��G�C���                                    Bx���  �          @���@���G����
�}�C��@��33�(���ٙ�C�9�                                    Bx��f  �          @��@����У׿�{���
C�s3@����(��.{��Q�C��                                    Bx��  �          @��@�33��(�����z�C�)@�33��\�0������C�ff                                    Bx��  �          @��H@�z��(���{�W�
C�#�@�z��
=q�����=qC�)                                    Bx�X  �          @�=q@�p���G���Q��e��C�>�@�p��   �
=����C��                                    Bx�)�  �          @�ff@��
��녿��H�n�HC���@��
��Q�G���33C�G�                                    Bx�8�  �          @�\)@��R��녿��R�s
=C��R@��R��ff�����8(�C���                                    Bx�GJ  �          @��@��;��þW
=��C�S3@��;Ǯ�����
C��                                    Bx�U�  �          @��@��H��Q쾸Q��l��C�=q@��H��z�>�ff@�33C�XR                                    Bx�d�  �          @��@���333��G���{C�` @���"�\?�
=A>ffC��f                                    Bx�s<  �          @��@��׿�@\)Aأ�C��\@��׿   @C�
B�C�޸                                    Bx���  	�          @���@����G�>�Q�@k�C�&f@����
=?�G�A&{C��                                    Bx���  �          @��@��
��(��u��HC�q@��
���?�@�=qC��H                                    Bx��.  T          @�(�@���ٙ�����33C�� @����G�>���@I��C��R                                    Bx���  T          @�{@�G��"�\>�(�@��HC���@�G��z�?��A|  C��R                                    Bx��z  T          @�@�ff�3�
?=p�@�
=C��
@�ff�
=q?�Q�A�  C��f                                    Bx��   �          @�33@��Ϳ�
=����z�C��{@��Ϳ��?�R@���C�K�                                    Bx���  �          @��@��ÿJ=q�����
=C��3@��ÿc�
�#�
��G�C�o\                                    Bx��l  �          @��\@�p��J=q�B�\��C��)@�p��J=q>8Q�?�C�ٚ                                    Bx��  �          @���?��H���Ϳ�����C���?��H��=q?�
=A��\C���                                    Bx��  �          @�?�33��{�:�H��z�C��?�33���?�z�A�33C�Ff                                    Bx�^  �          @�{?�����{���\�ffC���?�����G�?�33A�\)C�޸                                    Bx�#  �          @��R?�(����
��{�*�\C���?�(�����?��Ao33C��=                                    Bx�1�  �          @���@ff��ff�E���Q�C��f@ff���R?�G�A�G�C���                                    Bx�@P  �          @�
=?�z���(���  ��C�` ?�z���z�@�
A���C�1�                                    Bx�N�  �          @�(�?�(���  �z���{C��)?�(���?���A��HC�l�                                    Bx�]�  T          @��?�(����R��Q��^{C�ff?�(�����@\)A�C��                                    Bx�lB  �          @�?��H���R�(�����C�xR?��H����?�33A�{C���                                    Bx�z�  �          @��\?�
=��z�L���\)C�.?�
=��?ٙ�A�C�l�                                    Bx���  �          @�  ?�p���  �����3�C���?�p���p�?���Ai��C��
                                    Bx��4  �          @�z�?���������4  C���?������
?�=qA\(�C�˅                                    Bx���  �          @��@z���33��G���(�C���@z�����?L��A�HC�4{                                    Bx���  �          @�(�@.�R������(���\)C�5�@.�R���
>�p�@|��C�L�                                    Bx��&  �          @�(�@>�R��z�W
=�  C�
@>�R���?���AN{C�XR                                    Bx���  �          @��\@QG��\)����a�C��@QG�����?(�@��HC�!H                                    Bx��r  �          @��@G���(��G���p�C�c�@G���33<��
>k�C�"�                                    Bx��  �          @��
@HQ��j�H������C�AH@HQ�����>�?�z�C���                                    Bx���  �          @���@I�����\�8Q�����C��{@I���~�R?�z�AN=qC�0�                                    Bx�d  �          @���@!G������{�vffC�U�@!G����?&ff@��HC��q                                    Bx�
  �          @�
=@S33�q녿����p��C��=@S33�\)>�@���C���                                    Bx�*�  �          @�=q@<(��u��Ǯ��33C�Ǯ@<(����
>�\)@Mp�C�                                    Bx�9V  �          @���@H���g��\��ffC�xR@H���z�H>k�@'�C�Z�                                    Bx�G�  T          @���@K��hQ쿽p����HC��
@K��z=q>��@<��C���                                    Bx�V�  �          @�=q@Z=q�_\)����v{C�/\@Z=q�n{>��
@j=qC�AH                                    Bx�eH  �          @��@L���XQ��ff��p�C���@L���n{=�Q�?�  C�Z�                                    Bx�s�  �          @��H@K��N�R��Q���G�C�H�@K��h�ý��Ϳ��RC���                                    Bx���  �          @��\@j=q�QG���G��fffC��@j=q�_\)>�z�@Tz�C�.                                    Bx��:  �          @�(�@l���N{���R���HC�~�@l���c33=�\)?:�HC��                                    Bx���  �          @�(�@i���S�
��G��eG�C�޸@i���a�>��R@`  C��
                                    Bx���  �          @�{��=q�p�����C��{��=q��Q���Q��-G�C�Q�                                    Bx��,  �          @�33>�
=��G������
C�� >�
=��  ����9��C�                                    Bx���  T          @�z�@	����{���
�n��C���@	���r�\�e���C��H                                    Bx��x  �          @�{@&ff��
=���H�U  C��R@&ff�g��E��
ffC��)                                    Bx��  �          @���@6ff�����aG��8(�C��@6ff�L(�����ffC���                                    Bx���  �          @���@J=q����W
=�4��C�t{@J=q�(��#�
��C�,�                                    Bx�j  �          @�p�@��R�   ��
��  C��@��R�333��(��P(�C��                                    Bx�  �          @�@�
=��(���(����C��=@�
=�!G��5����C��                                    Bx�#�  �          @�p�@�  �(���{�i�C��)@�  �#�
��\)�AG�C�Ф                                    Bx�2\  �          @�=q@�ff��  �N�R��C�T{@�ff��{�4z�� {C��                                    Bx�A  �          @��\@�
=��\)�N�R�G�C�
@�
=����2�\��\)C�e                                    Bx�O�  �          @�(�@�����,������C�3@���ÿ�(����\C�R                                    Bx�^N  �          @��@�p��(���R���C�@ @�p���{��(���Q�C�u�                                    Bx�l�  �          @��
@��׽�G������p�C�]q@��׿���
�H����C�C�                                    Bx�{�  �          @��@�{�^�R������C���@�{�ٙ��Ǯ���C��                                    Bx��@  �          @�z�@��
����G���z�C��=@��
�
=��33�rffC�w
                                    Bx���  �          @�{@���\)�!G�����C�H�@�����ff��33C��                                    Bx���  �          @�{@�p����2�\���\C�<)@�p���ff�{��{C�>�                                    Bx��2  �          @��R@��;W
=�7
=��ffC��\@��Ϳ�33�   ��C��R                                    Bx���  �          @�\)@��H�\(��z���(�C��R@��H��G��޸R��33C��                                    Bx��~  �          @�p�@�  ��{�p����HC��@�  �
=q��
=�u�C��H                                    Bx��$  �          @���@��
���\)����C�Ff@��
��z��p���z�C�Ff                                    Bx���  �          @��\@��H�33��ff���RC�Q�@��H�.�R��(���
=C�G�                                    Bx��p  �          @��
@�  ��׿�ff�,��C���@�  �\)        C��f                                    Bx�  �          @��
@����\�Tz����C�=q@�����=���?�G�C�xR                                    Bx��  �          @�z�@������Q��n{C��@�����(���\)C��R                                    Bx�+b  �          @�=q@�(����H��{��(�C�@�(��   �Y���  C�Z�                                    Bx�:  	�          @�=q@�p���{���
�~�RC�� @�p���\)�Q��
=C�
=                                    Bx�H�  
�          @�=q@�\)��������e�C��R@�\)��\�333��Q�C���                                    Bx�WT  
�          @��@�zῴzῧ��\��C�Q�@�z��=q����ə�C�4{                                    Bx�e�  
�          @��
@�����Ϳ�33���C���@����@�׿5��z�C�*=                                    Bx�t�  �          @�Q�@�{�O\)������RC�%@�{�xQ�5�ᙚC��{                                    Bx��F  
Y          @���@�  �Z�H����33C��@�  ���\�333���C�>�                                    Bx���  �          @�\)@|(��c�
�Q����HC��3@|(����
����z�C�ٚ                                    Bx���  "          @�
=@}p��b�\�����
C��@}p���33����(�C�H                                    Bx��8  	�          @�p�@k��o\)�(����HC�Ff@k���녾�G����C�AH                                    Bx���  	�          @�  @�
=���
�����O\)C��3@�
=�녿���7
=C��                                    Bx�̄  
�          @�
=@��?c�
���\�W�A=q@��>�z���
��@HQ�                                    Bx��*  
(          @��@���?c�
�����3�
A(�@���>�p�����b=q@�                                      Bx���  
Z          @�Q�@�G�?����Q��	��AMG�@�G�?B�\���R�Qp�A
=                                    Bx��v  
�          @�{@�\)>Ǯ��33�E��@���@�\)��G����H�P  C�`                                     Bx�   "          @�33@��\�n{��Q��d(�C�AH@��\��Q�n{��RC���                                    Bx� �  �          @�=q@�
=��Q����uC�ٚ@�
=��(��k��G�C�H�                                    Bx� $h  	�          @��\@�z��
=��
=�c�C�S3@�z�����R��33C�B�                                    Bx� 3  �          @�33@�G���  ��(���33C���@�G��33�\(��
=C�>�                                    Bx� A�  �          @�(�@��R������3\)C��3@��R�����R�>�RC�e                                    Bx� PZ  "          @���@�zῆff����N�\C���@�z��  �E���C���                                    Bx� _   
Z          @���@�G���R����w
=C���@�G���Q쿔z��8��C���                                    Bx� m�  �          @��H@�\)�E���{��z�C��)@�\)���H����[�C���                                    Bx� |L  T          @���@�33�^�R��{�1C���@�33���R�+���z�C��                                     Bx� ��  "          @�=q@����(��\(��Q�C��
@����(���z��8��C��=                                    Bx� ��  
�          @��@��H��\)�z���{C��=@��H��(�=�?�Q�C�u�                                    Bx� �>  T          @���@�p��.{�8Q���{C��3@�p��h�þǮ�uC�l�                                    Bx� ��  T          @���@��
?u�(������Aff@��
?(��}p��\)@�                                    Bx� Ŋ  "          @�33@�ff>B�\�����>�R?�z�@�ff���R��
=�:�RC�s3                                    Bx� �0  �          @�@���>�  ����Hz�@(�@��þ�zΰ�
�G
=C���                                    Bx� ��  T          @�{@���>�33��(��=�@]p�@����#�
��  �C33C�0�                                    Bx� �|  "          @���@��=�Q쿢�\�G�?k�@����
=�����<��C��                                    Bx�! "  
�          @�=q@���>�(��&ff��{@���@���>��E���G�?��R                                    Bx�!�  �          @�ff@���>\)�
=��?���@��ͽ��Ϳ����Q�C��                                     Bx�!n  �          @�(�@���?�������G�A��@���?˅��  �D(�A�(�                                    Bx�!,  �          @��@�{@
�H>.{?��A�{@�{@�
�5��A��                                    Bx�!:�  
�          @�\)@��\?�{�B�\��A3�
@��\?p�׿(���{A��                                    Bx�!I`  �          @��@�p�?W
=����(�A
=@�p�?8Q��G���{@���                                    Bx�!X  �          @�=q@����#�
�W
=�
=C��@������;B�\���C��f                                    Bx�!f�  
�          @�{@�{<����
�aG�>���@�{<��
������>aG�                                    Bx�!uR  T          @�  @�
=>.{>�ff@��\?��@�
=>���>�33@U�@Mp�                                    Bx�!��  
�          @��@�Q�?O\)>�(�@�Q�A ��@�Q�?k�=��
?G�A�                                    Bx�!��  �          @�
=@��H?���>��@�z�A1p�@��H?��L�Ϳ   A=�                                    Bx�!�D  T          @�33@�ff?�(�>��
@EAA�@�ff?�  �B�\����AE�                                    Bx�!��  T          @�
=@���?���>.{?�{AV{@���?���\�l(�AO�                                    Bx�!��  �          @�
=@���?�p�>8Q�?�G�Ae�@���?�
=����y��A^ff                                    Bx�!�6  
�          @���@��?�ff>W
=@33Ao33@��?�G�����w
=Aip�                                    Bx�!��  T          @�33@��\?޸R�L�;�ffA��H@��\?˅�8Q���=qA~ff                                    Bx�!�  �          @��\@���?�Q�=u?#�
AeG�@���?��Ϳ   ���AW�
                                    Bx�!�(  T          @�G�@�z�?�  >��R@EAH(�@�z�?��\�L�Ϳ���AK�
                                    Bx�"�  �          @�  @��?�(�>�Q�@\��A;�@��?�G��\)���ABff                                    Bx�"t  T          @�  @�p�>�G�?.{@��@��@�p�?+�>�@�Q�@�                                    Bx�"%  "          @�z�@�=q>B�\?s33A��?�  @�=q?
=q?L��@�
=@�p�                                    Bx�"3�  "          @�p�@��H�L��?p��A�HC�)@��H>8Q�?p��A\)?�\)                                    Bx�"Bf  "          @�z�@��þ��?��AffC�� @��ü�?��A$��C��)                                    Bx�"Q  "          @�\)@�z�>#�
?z�HAG�?\@�z�?�?W
=@���@���                                    Bx�"_�  
(          @�p�@\>��H?c�
Ap�@��
@\?L��?#�
@�ff@�ff                                    Bx�"nX  "          @ə�@������@��A��\C�33@���c�
@=p�A�  C�^�                                    Bx�"|�  $          @��@�p���
@�RA�p�C���@�p��u@E�A�C�H                                    Bx�"��  	�          @ə�@��H��33@�A��RC�@��H�!G�@4z�A���C���                                    Bx�"�J  �          @ə�@�=q��33@�A�\)C��q@�=q�(�@7�A��C��                                    Bx�"��  �          @�  @�Q��z�@=qA�(�C��q@�Q��R@7
=Aڏ\C�Ǯ                                    Bx�"��  
�          @ƸR@�zῦff@p�A�G�C��f@�zᾸQ�@"�\A�p�C�&f                                    Bx�"�<  
�          @�G�@��\��\)@��A�33C��=@��\�
=@5�A�  C���                                    Bx�"��  
�          @�Q�@�{��z�@#�
A�C���@�{��@?\)A�C�                                      Bx�"�  
Z          @ə�@�
=��p�@��A���C��@�
=�n{@=p�A��\C�*=                                    Bx�"�.  �          @�  @��Ϳ�  @B�\A��C��H@��;�@^�RB�C�Z�                                    Bx�# �  "          @�
=@��R��p�@U�B�C�j=@��R��33@o\)B��C���                                    Bx�#z  �          @ƸR@�����
@@  A�=qC�b�@����@\��B��C��                                    Bx�#   T          @�{@��R��p�@5A�
=C��3@��R�
=q@Q�B G�C��                                    Bx�#,�  $          @ə�@��Ϳ��
@5AָRC�q@��;�Q�@L��A�
=C��                                    Bx�#;l  
�          @�33@�p���ff@!�A�=qC��=@�p���=q@5�A�Q�C���                                    Bx�#J  T          @�{@���G�@'�A�  C���@���(�@?\)A݅C��
                                    Bx�#X�  "          @��@�G���{@2�\A�Q�C��@�G���@L(�A�\C��)                                    Bx�#g^  "          @�(�@�(�����@)��Aď\C��@�(�����@>{A�Q�C�xR                                    Bx�#v  "          @���@��ÿ�ff@=qA���C���@��þ��
@.{A���C�g�                                    Bx�#��  "          @ə�@�p��ٙ�@�A�C��@�p��E�@)��A��HC��                                    Bx�#�P  T          @�=q@�G�����@A�\)C�j=@�G����H@��A��RC���                                    Bx�#��  
�          @˅@�����
=?��HA��
C��@����Y��@(�A��HC�ٚ                                    Bx�#��  V          @�=q@����\)@ffA�
=C�"�@���=p�@"�\A��
C�XR                                    Bx�#�B  	�          @��
@��H��G�@��A�  C��@��H��Q�@ ��A��C�:�                                    Bx�#��  �          @ȣ�@��H�Tz�@ffA�z�C���@��H�#�
@  A���C��R                                    Bx�#܎  �          @�p�@�{�L��@33A��C�/\@�{=���@�A��
?k�                                    Bx�#�4  �          @��@��H�z�?�A��C�G�@��H>#�
@   A�p�?�                                      Bx�#��  �          @�@�녿p��?�
=A��C��{@�녾8Q�@��A�\)C�'�                                    Bx�$�  �          @���@����G�@�RA�p�C���@����p�@"�\A�
=C�4{                                    Bx�$&  �          @�@�33��=q@�\A�
=C��R@�33���@'�A�=qC���                                    Bx�$%�  T          @θR@���G�@{A��C�  @���p�@!�A�ffC�33                                    Bx�$4r  T          @Ϯ@����\@�A���C���@���Q�@%A�=qC�=q                                    Bx�$C  �          @�\)@�ff���@�RA�G�C��H@�ff����@#33A�33C��                                    Bx�$Q�  �          @�
=@�\)���R@Q�A�C�#�@�\)�Ǯ@�A��RC�q                                    Bx�$`d  �          @�
=@�p����\@�A�
=C��@�p���p�@%A�Q�C�5�                                    Bx�$o
  �          @�  @�
=����@
�HA��
C���@�
=��\@ ��A�=qC��                                    Bx�$}�  �          @�ff@�����R@\)A��RC��@���
=@'�A�{C�"�                                    Bx�$�V  �          @�p�@��H����@�RA��RC�XR@��H�   @$z�A�p�C��\                                    Bx�$��  �          @�  @�녿�  @p�A�=qC��@�녿�@5�A�
=C�h�                                    Bx�$��  �          @�p�@�p���@
�HA��\C�g�@�p����
@��A��C�o\                                    Bx�$�H  �          @�
=@�(���ff@(�A�z�C���@�(��.{@&ffA���C��R                                    Bx�$��  �          @Ϯ@�z��=q@�A��C�Z�@�z῀  @#33A���C�1�                                    Bx�$Ք  �          @�\)@������@�A�{C�=q@�����\@#33A��C�{                                    Bx�$�:  �          @ҏ\@��ÿ���@{A���C���@��ÿs33@@  A�
=C�T{                                    Bx�$��  �          @�(�@���
=q@333A��
C�z�@�����@XQ�A��C���                                    Bx�%�  �          @�=q@�33�@.{AÅC��@�33��G�@Q�A�C��=                                    Bx�%,  �          @�z�@�\)�9��@�A�G�C�@�\)���@N�RA�G�C��                                     Bx�%�  �          @�G�@���H��@
=qA�{C���@����@G�A�C���                                    Bx�%-x  �          @�{@���S33?�  A|��C��@����R@2�\A�C�T{                                    Bx�%<  �          @У�@��\�0��?�(�A���C��{@��\��33@3�
A�  C��{                                    Bx�%J�  �          @У�@�����
?�(�A��RC��@��ÿ�p�@)��A��C��3                                    Bx�%Yj  �          @��H@�\)��=q@z�A��RC�y�@�\)��  @%�A�ffC�>�                                    Bx�%h  �          @�Q�@��ÿB�\@��A�G�C�c�@���=#�
@�A�
=>�
=                                    Bx�%v�  �          @�{@�ff�(��@ffA�G�C��3@�ff>B�\@�A�?�                                    Bx�%�\  �          @��@�(���ff@
=qA�{C�� @�(���@�RA�ffC���                                    Bx�%�  �          @�(�@�\)��p�?�{A���C�xR@�\)��(�@�A�=qC���                                    Bx�%��  
�          @�\)@�33�^�R?aG�AC��
@�33�>{@G�A�33C��f                                    Bx�%�N  �          @��@���x��>�@�\)C�P�@���_\)?��
A��HC���                                    Bx�%��  �          @�z�@�\)���?   @�{C���@�\)�n�R?��A��C��                                    Bx�%Κ  �          @���@��~{?n{A��C��@��Z�H@�RA�p�C�.                                    Bx�%�@  �          @�p�@����g
=?�\)A/
=C���@����@��@�A�C�9�                                    Bx�%��  �          @�z�@�z��S�
?�z�A_
=C�B�@�z��(Q�@��Ař�C�9�                                    Bx�%��  �          @�ff@�ff�I��?�A�{C��@�ff�@333A�ffC�Ǯ                                    Bx�&	2  �          @��
@�ff�A�?޸RA�z�C��R@�ff���@*�HA؏\C�&f                                    Bx�&�  �          @�Q�@�p��333?��A�z�C���@�p��   @-p�A���C�h�                                    Bx�&&~  �          @�=q@����/\)?�A��
C��@��׿�(�@(Q�A��C��=                                    Bx�&5$  �          @�  @���ٙ�?�G�AAC���@����
=?�G�A�=qC��                                    Bx�&C�  �          @���@�33��(�?p��A�C�Q�@�33���?��An=qC�Q�                                    Bx�&Rp  �          @�
=@��33?Y��A�HC�� @����?���Atz�C�j=                                    Bx�&a  T          @�z�@���*=q?���A;�
C�:�@���
=@G�A�z�C�Ǯ                                    Bx�&o�  �          @��\@���,��?�G�A Q�C��)@�����?���A��\C�=q                                    Bx�&~b  �          @���@�=q�8��?���A_\)C�!H@�=q�  @�A��\C�
=                                    Bx�&�  �          @���@����  ?���A}p�C��)@�������@�RA�ffC�33                                    Bx�&��  �          @��H@vff�U��   ����C���@vff�s33�G���C���                                    Bx�&�T  �          @�Q�@�(��>�R�L�Ϳ   C�H�@�(��5�?p��A\)C��                                    Bx�&��  �          @���@�=q�/\)>��R@C�
C��@�=q�\)?���A?\)C���                                    Bx�&Ǡ  �          @�Q�@�(��=p�?(��@�z�C�Z�@�(��%�?���A��C�\                                    Bx�&�F  �          @�G�@�Q��J�H?c�
Ap�C�*=@�Q��,��?��A��
C�5�                                    Bx�&��  �          @��@����C33?�33A8Q�C���@����   @A�C�%                                    Bx�&�  �          @��@���9��@%�AУ�C�Q�@����33@X��B  C�\)                                    Bx�'8  �          @�ff@���/\)=���?�  C�4{@���#33?}p�A�\C�H                                    Bx�'�  �          @��@���8Q�W
=���RC��R@���2�\?=p�@�z�C���                                    Bx�'�  �          @��@��H�6ff=��
?:�HC��{@��H�*�H?�  A(�C��R                                    Bx�'.*  �          @���@����3�
��
=���HC���@����2�\?�@���C���                                    Bx�'<�  �          @��@���� �׿������C�@����#33>���@@  C���                                    Bx�'Kv  �          @��@�z��333=L��>��C���@�z��(��?s33A��C�o\                                    Bx�'Z  �          @�=q@��8�ÿ���
=C��3@��9��>�G�@��
C���                                    Bx�'h�  �          @�Q�@�{��Ϳ�p��MC��@�{��R������C���                                    Bx�'wh  �          @�G�@�p��>�R>��@|��C�H@�p��,(�?���AP��C�>�                                    Bx�'�  �          @Å@�=q�S33?:�H@ۅC�w
@�=q�8��?�  A�p�C�%                                    Bx�'��  �          @�33@���R�\?O\)@�C�y�@���7
=?���A���C�B�                                    Bx�'�Z  �          @���@����J�H?�G�AC��q@����,(�?�(�A�ffC��                                    Bx�'�   �          @�G�@��R�S33?n{A�\C�,�@��R�5�?�Q�A��C�#�                                    Bx�'��  �          @��@���XQ�?�G�A?�
C���@���3�
@G�A�  C�                                      Bx�'�L  �          @��@���R�\?�{AO33C��@���,(�@�A���C��=                                    Bx�'��  �          @Å@��R�Tz�?�ffAD  C�R@��R�/\)@�A�p�C��=                                    Bx�'�  �          @��@��S33?�p�A;\)C��@��/\)@p�A�G�C�q�                                    Bx�'�>  �          @���@���N�R?���A]�C�T{@���'
=@��A�=qC�H                                    Bx�(	�  �          @�=q@�{�N{?�
=AY��C�u�@�{�'
=@�A��C�R                                    Bx�(�  �          @�  @����H��?�p�AdQ�C��=@����!G�@��A��
C�ff                                    Bx�('0  �          @�Q�@�p��6ff?�33A1�C��{@�p��?��RA�z�C�Ф                                    Bx�(5�  �          @�
=@���@��?��
AF=qC�t{@���p�@
=qA���C���                                    Bx�(D|  �          @�p�@�{�+�?Y��A(�C�T{@�{��\?��A�G�C��                                    Bx�(S"  �          @Å@�(��;�?�\)AuG�C�q@�(���@��A��HC�                                      Bx�(a�  �          @�G�@����>{?ǮAo\)C���@����@��A��C��f                                    Bx�(pn  �          @���@�\)�AG�?��Al��C�ff@�\)���@��A��
C�/\                                    Bx�(  �          @���@�Q��>{?�ffAm��C���@�Q��ff@��A���C�p�                                    Bx�(��  �          @���@��R�A�?�\)Ay��C�C�@��R�Q�@�RA�=qC�(�                                    Bx�(�`  �          @��@����HQ�?��
A�{C�` @�����@*=qAҸRC���                                    Bx�(�  �          @���@���L(�?�33A�p�C��@���(�@2�\A�
=C�Y�                                    Bx�(��  �          @�\)@����0��?�\)A}G�C��)@�����@��A��C��f                                    Bx�(�R  �          @��@�G��&ff?�33A��C�\)@�G���(�@�A���C�T{                                    Bx�(��  �          @��R@�=q�-p�?���At��C���@�=q�ff@�A��
C��q                                    Bx�(�  �          @��H@�{�(��?��A��HC��@�{� ��@�A��
C���                                    Bx�(�D  �          @��R@�{�+�?���A�p�C�&f@�{��p�@%�A��HC���                                    Bx�)�  �          @�  @�ff�7
=?�A���C�W
@�ff�{@p�Aʏ\C�^�                                    Bx�)�  �          @��\@�33�333?���Az�\C��q@�33�(�@ffA�Q�C��{                                    Bx�) 6  �          @�=q@���-p�?��
AK\)C���@�����@�\A�Q�C�)                                    Bx�).�  T          @�(�@���$z�?�=qAP��C���@���33@�\A���C�\                                    Bx�)=�  �          @��@���+�?�(�AfffC�
=@���
=@p�A��C��=                                    Bx�)L(  �          @��@�
=�,(�?��
Aq�C���@�
=�
=@G�A�=qC�~�                                    Bx�)Z�  �          @�=q@����{?�=qA{�
C�� @��׿��@  A��C���                                    Bx�)it  �          @�  @�=q�{?���A/�
C���@�=q��?��A��HC�{                                    Bx�)x  �          @�@����?�=qA0Q�C��@�����z�?޸RA�z�C���                                    Bx�)��  �          @�p�@��H�(�?��RAIG�C�T{@��H��p�?�=qA��HC��H                                    Bx�)�f  �          @���@���(�?�(�AG
=C�K�@�녿�p�?�A��
C���                                    Bx�)�  �          @�z�@�Q��
=?��A*ffC�h�@�Q��Q�?�Q�A�  C�k�                                    Bx�)��  T          @��
@�����?��RAK�C��)@�����z�?�A�z�C��                                    Bx�)�X  �          @��@�  �ٙ�?uAffC��@�  ���?�Ak�C���                                    Bx�)��  �          @�p�@�{����?��AR�HC�0�@�{���?��
A��\C��                                    Bx�)ޤ  T          @��R@�z����?�Q�A@��C�k�@�z��  ?��A��
C���                                    Bx�)�J  �          @��@��H�0��?���A^{C�!H@��H�\)@Q�A�(�C��R                                    Bx�)��  �          @�\)@����=q?��
AN�\C�1�@��ÿ�
=?�
=A��C��H                                    Bx�*
�  �          @��@��H��\?��AXQ�C���@��H���?���A�
=C�L�                                    Bx�*<  �          @��@�=q�33?��A_�C�˅@�=q��ff@   A�
=C�@                                     Bx�*'�  �          @�=q@����ff?�Q�Aq��C��3@�����=q@�
A�33C���                                    Bx�*6�  �          @���@����@�\A�p�C��@��=�Q�@ffA�p�?��                                    Bx�*E.  �          @��
@����R@�HA�Q�C�!H@�>��@��A���@�                                      Bx�*S�  �          @���@��
?Y��@1G�A�33A1�@��
?�\)@��A�Q�A���                                    Bx�*bz  �          @�\)@�z�Ǯ?h��A=qC��3@�z῝p�?���A`��C�H�                                    Bx�*q   �          @�  @�33���?��A]G�C�� @�33���?ٙ�A�Q�C���                                    Bx�*�  �          @�p�@��Ǯ?���A���C�9�@����?���A��C��                                    Bx�*�l  �          @�{@�\)��  ?\A�Q�C���@�\)��  ?��A���C�S3                                    Bx�*�  T          @�z�@�ff��ff?У�A���C���@�ff�G�?�Q�A�33C���                                    Bx�*��  
�          @�@��\���?�A���C��{@��\���R@
=A���C��\                                    Bx�*�^  �          @��
@�33�333?�z�Au�C��@�33��=q?�ffA�G�C�}q                                    Bx�*�  "          @�G�@��׾���?�Q�A
=C���@���<��
?��RA���>�                                      Bx�*ת  �          @�{@��H�
=q?�ffA��HC��\@��H���
?��A�\)C���                                    Bx�*�P  
Z          @��@�33�Ǯ?�=qA��C��H@�33��ff?���A��
C��3                                    Bx�*��  T          @�
=@�����?���A�z�C��\@�����=q@{AʸRC�K�                                    Bx�+�  
�          @���@����  @��A��C���@���Q�@!�A�p�C��)                                    Bx�+B             @��H@��H�˅@Q�A�Q�C��@��H�n{@   A���C��R                                    Bx�+ �  
�          @�33@�=q���@ffAۅC�XR@�=q�!G�@(Q�A���C�ٚ                                    Bx�+/�  
�          @��H@���O\)@��A�z�C��f@����G�@!G�A�{C�J=                                    Bx�+>4  
�          @�G�@��G�@�
AϮC�@ @�����@(�A�  C�]q                                    Bx�+L�  �          @��
@�p���  ?�33A���C��f@�p��:�H?�
=A�{C���                                    Bx�+[�  T          @��R@��Ϳ�?�\)As�C��f@��Ϳ��H?���A�Q�C�C�                                    Bx�+j&  �          @���@��R�p�?���Ax��C�h�@��R��(�@ ��A��\C��{                                    Bx�+x�  "          @���@����?��HAz�HC�/\@��׿���?�p�A�(�C��)                                    Bx�+�r  �          @���@�{��
?ٙ�A��HC�#�@�{��G�@��A��HC�
                                    Bx�+�  T          @�Q�@�p���
?��
A��HC���@�p���p�@ffA�(�C��R                                    Bx�+��  �          @�Q�@�33��@G�A�=qC�Ф@�33�У�@$z�Aޣ�C�<)                                    Bx�+�d  �          @��\@����\)?��A�p�C�w
@��ÿ�z�@�A��C�p�                                    Bx�+�
  "          @��\@�녿���@�
A�z�C��3@�녿aG�@*=qA�(�C��R                                    Bx�+а  T          @��H@�G����H@�A��HC���@�G�����@"�\A�ffC�O\                                    Bx�+�V  �          @�Q�@��H��@2�\A��C�޸@��H��Q�@@  Bp�C��f                                    Bx�+��  
�          @��R@��ÿ��@+�A��
C�w
@��ÿ(�@=p�B��C�1�                                    Bx�+��  
�          @��@��ÿ���@1�A���C���@��ÿ@  @FffB�C��                                    Bx�,H  �          @�33@�G���{@N�RBC��R@�G���
=@^�RB"\)C�                                      Bx�,�  T          @�=q@��׿�=q@Mp�Bz�C��R@��׾��@\��B!��C�R                                    Bx�,(�  
�          @���@x��>k�@Mp�B
=@Tz�@x��?��@A�B�\Az=q                                    Bx�,7:  �          @��R@��\��@X��B
=C�P�@��\>��@Z=qB��@�                                      Bx�,E�  �          @���@E��Z�H@A�B�\C��@E��{@uB0�HC���                                    Bx�,T�  "          @��
@Vff�n{?��RA�Q�C��R@Vff�K�@(�A�ffC�5�                                    Bx�,c,  �          @�z�@s33@=q@:�HB{B�@s33@G
=@
�HA�z�B=q                                    Bx�,q�  T          @��R@�����@�RA�  C��H@��333@ ��A�=qC�|)                                    Bx�,�x  "          @��\@��\�6ff?�z�A���C�XR@��\��R@&ffA�RC�`                                     Bx�,�  �          @�ff@�G��5�?��
A��RC�E@�G��  @{A�=qC�"�                                    Bx�,��  
�          @�@�Q��#�
@
=A��C�� @�Q��33@-p�A�\)C��{                                    Bx�,�j  
'          @���@��
�/\)@��A��RC�+�@��
��@2�\A�G�C���                                    Bx�,�  "          @�=q@����
=q@33A͙�C��3@��׿�(�@333A�  C�~�                                    Bx�,ɶ  
(          @��@�
=��G�@
=A�33C�C�@�
=����@/\)A�z�C�G�                                    Bx�,�\  �          @��\@�p��s33@ffA��C�>�@�p���z�@!G�A�=qC�1�                                    Bx�,�  �          @��
@�Q�޸R@  A�p�C�j=@�Q쿋�@(Q�A�Q�C�4{                                    Bx�,��  �          @��\@�33�\(�?��A�C��@�33��{@33A��C��=                                    Bx�-N  �          @���@�=q?Tz�?��AHz�A�@�=q?���?aG�AffADz�                                    Bx�-�  �          @���@�33?�  ?
=q@��HA\Q�@�33?���>B�\@�
Al��                                    Bx�-!�  �          @�G�@�\)>�33>�Q�@|(�@vff@�\)>�(�>��@2�\@�                                      Bx�-0@  �          @��\@�G�>B�\��Q�k�@ ��@�G�>#�
����?ٙ�                                    Bx�->�  �          @�=q@��\�Q�   ���
C��@��\�(�=��
?Y��C��\                                    Bx�-M�  �          @���@�Q�?n{�B�\�ffA(z�@�Q�?Y����
=��Q�A{                                    Bx�-\2  �          @�=q@���=�G�>aG�@ff?���@���>#�
>8Q�?���?�\                                    Bx�-j�  �          @�z�@�\)����?�G�A0��C�k�@�\)���?�=qAi��C��                                    Bx�-y~  �          @��@�Q�?L��>���@eA	�@�Q�?\(�=�?��RA                                      Bx�-�$  T          @��@�=q��
=�L�Ϳ�C���@�=q��
==u?(��C��q                                    Bx�-��  �          @��\@�G��\>��@���C��\@�G���  ?��@�ffC���                                    Bx�-�p  �          @�
=@�Q쿏\)?Y��A�\C��@�Q�^�R?�{AEp�C��                                    Bx�-�  �          @�  @�  <#�
?�\@�33>��@�  >\)>��H@��?�(�                                    Bx�-¼  �          @�=q@���?�����Ϳ�{A7
=@���?z�H��p��\)A,z�                                    Bx�-�b  �          @���@�\)?�\>�{@i��@���@�\)?z�>B�\@
=@ə�                                    Bx�-�  �          @��R@��?+�>#�
?�G�@�z�@��?0�׼��
�8Q�@�33                                    Bx�-�  �          @���@��R?z�>���@R�\@���@��R?#�
>\)?�ff@��                                    Bx�-�T  �          @�
=@�{>�>k�@�R@��\@�{?   =���?�{@�G�                                    Bx�.�  �          @�ff@�{�#�
>.{?��C���@�{<�>.{?�\)>�{                                    Bx�.�  �          @�  @�\)=�=�G�?��R?�\)@�\)>��=��
?Y��?��                                    Bx�.)F  �          @�
=@�{>����L�Ϳ�\@S33@�{>�\)����=q@C�
                                    Bx�.7�  �          @��@�
=�8Q�>�?�
=C�f@�
=�\)>.{?��C�=q                                    Bx�.F�  �          @���@�(�����=�\)?L��C�ff@�(���33>\@��\C��{                                    Bx�.U8  �          @�G�@�����>�
=@�Q�C��@�����?:�H@���C��R                                    Bx�.c�  �          @�  @��\��p�>�Q�@y��C�0�@��\����?(��@�C��                                    Bx�.r�  �          @�Q�@����p�?.{@�\)C�� @����G�?��
A5��C��                                    Bx�.�*  �          @��@��(��>�p�@�33C�]q@����?�@�G�C��q                                    Bx�.��  �          @�ff@��Ϳ!G�>W
=@z�C���@��Ϳ\)>�Q�@\)C��=                                    Bx�.�v  �          @�@�(��논����RC��3@�(��\)=�?��
C��                                    Bx�.�  �          @�\)@��ÿu?h��A ��C���@��ÿ5?�\)AF�RC��                                    Bx�.��  �          @��@����B�\?���AHQ�C���@�����ff?��Ad  C�q�                                    Bx�.�h  �          @��@�G����?h��A"�HC��@�G�����?��\A6ffC�Ff                                    Bx�.�  �          @�{@�33�333>��?�\)C�
@�33�#�
>��R@`��C�h�                                    Bx�.�  �          @��H@�\)�s33=�?���C��)@�\)�c�
>�Q�@uC�.                                    Bx�.�Z  �          @�{@�ff���?��Al(�C���@�ff�W
=?�z�A}�C���                                    Bx�/   �          @�@�z�>W
=�B�\��@33@�z�>���u�'�?��H                                    Bx�/�  �          @�z�@�
=>#�
���H�Z�H?�@�
=������H�[33C�(�                                    Bx�/"L  �          @�z�@�\)=��
�aG��\)?c�
@�\)�\)�^�R��C�1�                                    Bx�/0�  "          @��H@��?����R�e�A�G�@��?�ff�&ff���AuG�                                    Bx�/?�  �          @��
@�p�@%���
�W
=A�
=@�p�@   �&ff��\A�z�                                    Bx�/N>  �          @�33@���@p��L�Ϳ\)A�(�@���@Q�
=��ffAə�                                    Bx�/\�  �          @�Q�@��?����=q�x  A��
@��?xQ��\)���A?33                                    Bx�/k�  �          @���@��?�p��W
=�=qA��\@��?�(����R�c�A��                                    Bx�/z0  �          @��@�
=?�
=�@  ���A��@�
=?�������PQ�A���                                    Bx�/��  �          @�\)@�G�?�\)��33��ffA�
=@�G�?�{�\)��z�Ag
=                                    Bx�/�|  T          @�\)@�=q?�\)����A�A��@�=q?��ÿ�33����A�ff                                    Bx�/�"  �          @���@�\)?�=q�n{�+\)A{�@�\)?��ÿ�(��b�RAK�                                    Bx�/��  �          @�  @�33?�녿@  ��A�
=@�33?�z῏\)�RffA�ff                                    Bx�/�n  �          @�@�Q�?У׿h���,Q�A�\)@�Q�?�\)���
�s33A���                                    Bx�/�  �          @��R@�ff@����\�?
=A뙚@�ff@�
�Ǯ��AЏ\                                    Bx�/�  �          @��R@`��@G�����yB&ff@`��@-p��G���(�B�                                    Bx�/�`  �          @��@X��@5��������HBz�@X��@���R���B��                                    Bx�/�  �          @�@����Ǯ�p���)G�C���@����޸R�
=q��G�C��                                    Bx�0�  �          @��@���Q��p��ʣ�C���@���
=��ff���C�:�                                    Bx�0R  �          @�ff@���Y�������G�C��f@����z��
�H��=qC�J=                                    Bx�0)�  �          @��@��ÿ+��2�\�  C���@��ÿ����"�\����C�l�                                    Bx�08�  �          @���@��׿������ffC��@��׿����33���C��                                     Bx�0GD  �          @�  @��Ϳ}p������C���@��Ϳ��H�����\)C��3                                    Bx�0U�  �          @��R@��׿��\��  ���C�&f@��׿˅��33�X��C�O\                                    Bx�0d�  �          @�(�@��Ϳ�\)���H����C���@��Ϳ��R��33��C���                                    Bx�0s6  �          @���@��k��B�\��C�n@���=q����ffC��f                                    Bx�0��  �          @��R@��Ϳ��
����r�HC�� @��Ϳ����G��=p�C�%                                    Bx�0��  �          @���@��;�  ��{��  C�p�@��Ϳ.{�޸R����C��\                                    Bx�0�(  �          @�ff@��>��������@��@����\)�\)���
C���                                    Bx�0��  �          @��@��?��$z�� \)A z�@�녽L���(Q���HC���                                    Bx�0�t  �          @�p�@c�
?�R� ���
�
A{@c�
<#�
�%���>u                                    Bx�0�  �          @�Q�@   @6ff�e�7�\B[��@   ?�
=����a{B0
=                                    Bx�0��  �          @�Q�@�@��c�
�;�
B2
=@�?�
=��Q��]ffA�
=                                    Bx�0�f  �          @�Q�@�
=��zᾏ\)�C�
C�b�@�
=��33�L���{C�3                                    Bx�0�  �          @�  @���aG�>aG�@��C�&f@���O\)>�
=@��\C��\                                    Bx�1�  �          @�\)@�(��Tzᾨ���l(�C�g�@�(��c�
������C��                                    Bx�1X  T          @�Q�@�����Y���z�C�#�@���333�333��{C�"�                                    Bx�1"�  �          @���@�p����ÿk��!�C�0�@�p����Q��33C�\                                    Bx�11�  �          @��H@�����Q���C�#�@���333�.{�陚C�0�                                    Bx�1@J  �          @��\@�ff�5�aG��z�C�
@�ff�fff�0����RC�R                                    Bx�1N�  �          @�33@�녾.{�����ffC��@�녾�z�   ���C�k�                                    Bx�1]�  �          @�z�@��\�����333��C�]q@��\���(�����C���                                    Bx�1l<  �          @�z�@��
��Q쾳33�n{C�� @��
�.{���
�X��C��                                    Bx�1z�  �          @�p�@�z��G�����33C�p�@�z�aG����H��{C��                                    Bx�1��  �          @�{@�z�5�B�\��\C�AH@�z�:�H����=qC�                                      Bx�1�.  �          @�{@���h�ý����C�+�@���k�=�Q�?s33C�&f                                    Bx�1��  �          @��@��
=��
�\���H?Tz�@��
�#�
�Ǯ��p�C��\                                    Bx�1�z  �          @�\)@�����
�L�Ϳ
=qC��@����  >���@H��C��                                    Bx�1�   �          @�{@���>�{���H�|��@u�@���    ��  ���<#�
                                    Bx�1��  �          @��@�z�?B�\�ٙ���33A@�z�>�33��=q��z�@vff                                    Bx�1�l  T          @�\)@�z�?��������RA\z�@�z�?.{��ʸR@��R                                    Bx�1�  T          @�G�@��R?ٙ����ÅA��\@��R?����'
=���AXz�                                    Bx�1��  �          @�=q@��
?��H�����{A^�R@��
?!G��'
=�ߙ�@��
                                    Bx�2^  "          @��@�\)?�G��'���
=An�H@�\)?!G��5���@��
                                    Bx�2  �          @���@��\?}p��#�
�ۅA8��@��\>\�.{��=q@�                                      Bx�2*�  �          @���@���?���!���p�AT��@���?��.�R��p�@��                                    Bx�29P  �          @���@��?����)����ffAQG�@��>���5�����@�ff                                    Bx�2G�  �          @���@���?���� ����ffAIG�@���>���,(����@���                                    Bx�2V�  �          @���@�  ?���#33����Ar�R@�  ?.{�1G���  A{                                    Bx�2eB  �          @���@�(�?��\�-p��뙚At��@�(�?�R�;�� =q@���                                    Bx�2s�  �          @�\)@�
=?�ff�%���A�33@�
=?��:�H�Q�Aj=q                                    Bx�2��  �          @�\)@�z�?+��c�
��A
�H@�z����g��
=C�f                                    Bx�2�4  �          @�{@�G�?=p��Q���{A�@�G�>8Q���R�х@�                                    Bx�2��  T          @�\)@�zᾀ  �����ffC��3@�z�(�ÿ�����C�}q                                    Bx�2��  �          @��\@��=�G�����Z{?�Q�@���B�\��ff�X��C�                                    Bx�2�&  �          @���@��?
=q�#�
�Ǯ@���@��?��#�
�У�@�=q                                    Bx�2��  �          @��@��R?333?�Q�AB�R@�G�@��R?s33?�  A#�
AG�                                    Bx�2�r  �          @�ff@��R?}p�?��\AN�RA$Q�@��R?�  ?��\A%�AN=q                                    Bx�2�  �          @��
@�G�?��R?�z�A?
=A{�@�G�?��H?L��A
=A�p�                                    Bx�2��  �          @��
@��?��?(�@�  A�
@��?У�>�  @%�A���                                    Bx�3d  �          @���@�
=>�����@���@�
=>��ÿ(���33@\(�                                    Bx�3
  �          @��R@��H>�������5�?�  @��H��G������6�\C�s3                                    Bx�3#�  �          @�
=@��
�#�
����'�
C��=@��
��=q��G��"{C��q                                    Bx�32V  T          @�{@�=q�L�Ϳ���1C���@�=q��zῇ��+�C��H                                    Bx�3@�  �          @�p�@�33?p�׿���f{A�@�33?(�������@Ϯ                                    Bx�3O�  �          @���@���?=p���z��g�@�=q@���>�녿���}p�@�33                                    Bx�3^H  �          @�@�
=>8Q쿌���5�?�{@�
=��\)��{�8  C��H                                    Bx�3l�  �          @�Q�@��ÿk��z�����C��@��ÿ�{����
=C���                                    Bx�3{�  �          @�  @�G��}p�� ������C���@�G���33�޸R��
=C���                                    Bx�3�:  �          @�
=@�ff��������C�Y�@�ff���R��{����C�f                                    Bx�3��  �          @��@��R��33�����Q�C�|)@��R���
���
�x��C���                                    Bx�3��  �          @��R@�33�33��Q����C�S3@�33�*�H����a�C���                                    Bx�3�,  T          @�p�@����ff�
=q��(�C�)@����!녿�33���C��                                    Bx�3��  �          @���@�G��{�{��{C��
@�G��-p�����p�C���                                    Bx�3�x  �          @�  @X����=q��(���=qC��f@X����33�J=q�{C��R                                    Bx�3�  T          @�(�@,�����׿�{�W
=C��@,����ff�k��p�C��
                                    Bx�3��  T          @�@@  ��(������u�C���@@  �������ffC�7
                                    Bx�3�j  T          @�ff@=q��  �����M�C�T{@=q����\)����C�                                      Bx�4  �          @�G�@���
�(����Q�C�7
@��(�?
=@�33C�4{                                    Bx�4�  �          @\?���  ��ff��  C�7
?����R?Q�@��C�Ff                                    Bx�4+\  �          @�  @
=��  ������\C�"�@
=��\)?8Q�@�33C�+�                                    Bx�4:  �          @��@{����=p���Q�C�Y�@{��(�?\)@���C�Q�                                    Bx�4H�  �          @\@�
��Q쿜(��8��C�]q@�
��z�<#�
=�Q�C�                                      Bx�4WN  �          @�Q�@G���(���{�%G�C���@G���\)>��?�\)C��f                                    Bx�4e�  �          @���@�H��
=��
=�{�C��{@�H��ff��ff��p�C���                                    Bx�4t�  �          @��
@.�R��G��Z=q���C�f@.�R��{�����=qC�q                                    Bx�4�@  T          @��
@:�H�g��|���!\)C�q�@:�H�����Dz���p�C��                                    Bx�4��  T          @�\)@]p��mp��E���\C�� @]p���G��p����\C�o\                                    Bx�4��  �          @���@S33�AG������'�
C���@S33�tz��QG���RC�k�                                    Bx�4�2  �          @�G�@G
=�Dz���z��-��C��@G
=�y���XQ��\)C�L�                                    Bx�4��  �          @���@J�H�B�\����,G�C�R@J�H�w
=�Vff�ffC��3                                    Bx�4�~  �          @���@h���!����\�*��C���@h���Vff�[���C��f                                    Bx�4�$  �          @�G�@q��,���vff���C�=q@q��^{�K���  C���                                    Bx�4��  �          @���@��ÿ�G��u��&C��3@�����
�Z=q�Q�C�R                                    Bx�4�p  �          @�(�@n�R���~�R�+�HC�g�@n�R�8���[���C�
                                    Bx�5  �          @�Q�@]p��\)��ff�2=qC�!H@]p��U�c�
�{C��                                    Bx�5�  �          @�G�@e�z���Q��3��C���@e�K��j=q�G�C�>�                                    Bx�5$b  �          @�{@����*=q�z�H�#\)C�g�@����3�
�Ǯ�\)C���                                    Bx�53  �          @�G�@�33�^{?#�
@ÅC�0�@�33�P  ?�\)AS�C��                                    Bx�5A�  �          @Å@��H�J=q?��A$z�C��@��H�6ff?޸RA���C�b�                                    Bx�5PT  �          @�p�@��O\)?��
A�  C�^�@��1�@�A��C�O\                                    Bx�5^�  �          @�(�@�Q��k�?s33A(�C�"�@�Q��X��?�(�A��C�C�                                    Bx�5m�  �          @�(�@�\)�l��?�A0(�C���@�\)�W
=?�Q�A��C�K�                                    Bx�5|F  �          @��H@����tz�?��HA8��C�H@����^{@ ��A��\C�Y�                                    Bx�5��  �          @�33@���_\)?��AQ�C�  @���G
=@ffA�\)C���                                    Bx�5��  �          @�=q@�{�K�?��RAc
=C��H@�{�1�@Q�A���C�N                                    Bx�5�8  �          @�G�@�  �U?�ffAo
=C�w
@�  �;�@�RA�=qC�0�                                    Bx�5��  �          @���@����2�\?�A���C�xR@����@
=A���C��                                    Bx�5ń  �          @���@��� ��?���A�\)C�
=@���33@ffA��C�(�                                    Bx�5�*  �          @�\)@���Q�?��A�
=C��\@����
=@�\A���C��                                    Bx�5��  �          @�
=@�G�����?��HA��C���@�G���p�@A�=qC�0�                                    Bx�5�v  �          @�Q�@�녿�\)@
=qA�
=C�"�@�녿�\)@p�A���C��f                                    Bx�6   �          @�z�@�G�=#�
?���A��
>�
=@�G�>�(�?��A�\)@�                                    Bx�6�  �          @�(�@�33�z�?�\)A��
C��@�33�aG�?��HA�G�C��\                                    Bx�6h  �          @�=q@�  ��
=?�ffA��C��R@�  ���
?���A�G�C�޸                                    Bx�6,  �          @��\@������?�{A��C��3@��<#�
?�33A�G�=��
                                    Bx�6:�  �          @���@�����
=?��A���C��)@���    ?�Q�A���C���                                    Bx�6IZ  �          @��
@��þW
=?�{A��HC��H@���>B�\?�{A���@
=                                    Bx�6X   �          @��
@��
�Tz�?�Q�Amp�C�� @��
�   ?�=qA��HC�W
                                    Bx�6f�  �          @�
=@�z΅{?��
AN�\C���@�zῇ�?��Ay�C�q�                                    Bx�6uL  �          @�G�@����=q?��
Aup�C��@���z�H?��
A�\)C��                                    Bx�6��  �          @�  @�ff��=q?�Q�Ah��C�ff@�ff�@  ?��A���C��                                    Bx�6��  �          @��@��ÿ�Q�?��HAo�C�^�@��ÿ��?޸RA���C�%                                    Bx�6�>  �          @��R@�p����?�Q�AAG�C�7
@�p����\?���Aj�HC���                                    Bx�6��  �          @�
=@�����?�Q�A@z�C�ff@���O\)?�33Ab=qC�Ф                                    Bx�6��  �          @�ff@����\?�\)A_33C��3@��333?ǮA~{C�S3                                    Bx�6�0  T          @�p�@�����?��A,��C�=q@�����z�?��Ac�
C��\                                    Bx�6��  �          @��@�ff��\?z�HA!p�C���@�ff�\?�=qA\��C�ٚ                                    Bx�6�|  �          @��H@��R��\?xQ�A�C��@��R�\?���A[
=C��H                                    Bx�6�"  �          @��
@�ff��ff?�\)A7�C�xR@�ff�\?�(�As�C�޸                                    Bx�7�  �          @�33@�\)��G�?uA=qC���@�\)��G�?��AX��C��{                                    Bx�7n  �          @��\@�\)��z�?�G�A'\)C�1�@�\)��z�?��A^�HC�y�                                    Bx�7%  �          @��@�
=��\?�p�AIG�C�� @�
=��(�?�=qA��C�#�                                    Bx�73�  �          @��R@�G����H?�  AS\)C���@�G���33?�=qA���C�9�                                    Bx�7B`  �          @��@�\)�У�?��A`  C��\@�\)����?�\)A��C���                                    Bx�7Q  �          @�p�@�z�޸R?��RA�ffC�7
@�z῰��?�=qA�ffC��                                    Bx�7_�  �          @�@�(���(�?��\AX��C��@�(���33?�z�A���C���                                    Bx�7nR  �          @���@�����
?���A<  C��)@�����\?��Ag33C�b�                                    Bx�7|�  �          @�\)@����(�?��A/33C��@����(�?�=qA`��C�\)                                    Bx�7��  �          @�Q�@�����\?�p�AO
=C�Y�@�����(�?�=qA���C��                                    Bx�7�D  �          @���@�=q���?�G�AS
=C�H�@�=q��p�?�{A�C�޸                                    Bx�7��  �          @�G�@�(���ff?��
A,  C�Z�@�(����
?��Ah��C���                                    Bx�7��  �          @�=q@��ÿ�Q�L�Ϳ\)C�XR@��ÿ�>��@+�C�t{                                    Bx�7�6  �          @�33@�33��\�\�}p�C�E@�33�Q녿����d��C���                                    Bx�7��  �          @�(�@��Ϳ���Y���  C�Ff@��Ϳ�Q�\)��
=C���                                    Bx�7�  �          @�{@�  ��{<�>�=qC��@�  ����>��
@L��C�<)                                    Bx�7�(  �          @�z�@�p����׿&ff���C���@�p����R��{�`��C�S3                                    Bx�8 �  �          @�@�ff��33�+���G�C�w
@�ff��G���33�n{C��                                    Bx�8t  �          @�{@�ff�@  ����{C�33@�ff��(��z�����C�U�                                    Bx�8  T          @��@Z�H�E��P  �+�C���@Z�H��Q��AG��p�C��\                                    Bx�8,�  �          @�
=@�Q쿮{��R��z�C���@�Q��������
C��                                    Bx�8;f  �          @�\)@[��h���s33�<{C��\@[��ٙ��aG��*��C��\                                    Bx�8J  �          @�G�@l(��s33�j�H�/�
C���@l(����H�XQ���C���                                    Bx�8X�  �          @��@w���
=�e�&�\C���@w����P���=qC��                                    Bx�8gX  �          @���@�  �˅�H���
�C���@�  �{�.�R��
=C�/\                                    Bx�8u�  �          @��@�z��ff�7
=��=qC��q@�z����=q��  C��R                                    Bx�8��  �          @�z�@����p�������C�1�@�����H>���@Mp�C�J=                                    Bx�8�J  �          @��\@���������C��{@����
>�p�@s�
C��3                                    Bx�8��  �          @��H@�p��Q�>�@�ffC�G�@�p���p�?k�AG�C���                                    Bx�8��  T          @��\@�ff�33>L��@ ��C��@�ff���H?#�
@���C��                                    Bx�8�<  �          @�33@��׿�=�\)?0��C�o\@��׿�{>�@�{C���                                    Bx�8��  T          @���@�{��Q�\)��33C�33@�{��>�\)@/\)C�C�                                    Bx�8܈  �          @��R@����	�������W
=C��@����
�H>�?�=qC���                                    Bx�8�.  
�          @��@����=q�L�Ϳ�(�C��3@������>��
@J=qC�                                    Bx�8��  �          @�z�@�(��33��z��:=qC���@�(���
>W
=@
�HC��                                     Bx�9z  �          @���@��#33=#�
>�p�C��@��\)?z�@���C�/\                                    Bx�9   �          @��@�=q�#33��33�Y��C�0�@�=q�$z�>L��?��HC��                                    Bx�9%�  �          @��
@�  �'�>.{?�\)C��)@�  �!G�?8Q�@�=qC�'�                                    Bx�94l  �          @���@����Q콏\)�(��C��@����>�G�@���C�G�                                    Bx�9C  �          @�\)@��R�=q��z��5�C��@��R��H>k�@(�C��                                    Bx�9Q�  �          @���@�(�����\�#�C���@�(��   ���H��ffC�                                      Bx�9`^  �          @�ff@��
��R��33�9C�@ @��
��H�#�
��\)C�Y�                                    Bx�9o  �          @��R@�������+�C�޸@��33�z����
C��                                    Bx�9}�  �          @��@�G������\)��z�C�w
@�G��녿����33C�B�                                    Bx�9�P  �          @��\@�\)�   ��z��k33C��@�\)��׿s33��C��                                    Bx�9��  �          @��H@�����R��G��'�C�` @���
=q�\)���C���                                    Bx�9��  �          @�Q�@�
=��>\)?�G�C���@�
=���?!G�@ӅC�{                                    Bx�9�B  �          @��R@����  ?   @�33C��@����?xQ�A&{C�w
                                    Bx�9��  �          @�33@���\)�p����
C�{@��녿����C�O\                                    Bx�9Վ  �          @�Q�@������R�(���\)C��@����޸R����33C��                                    Bx�9�4  �          @��@���33�
�H��ffC�k�@�����������C�<)                                    Bx�9��  �          @��\@����������\)C���@����G���\)��\)C��)                                    Bx�:�  �          @��@�녿�G���(�����C�3@�녿�녿����|��C�,�                                    Bx�:&  �          @���@��Ϳ��H�������C�t{@��Ϳ�=q�\�o\)C���                                    Bx�:�  �          @�{@�  �������Q�C�g�@�  ��z�\�l��C��                                     Bx�:-r  �          @�p�@�ff���Ϳ����\C�J=@�ff���R�����p�C�\)                                    Bx�:<  �          @�\)@�
=��  ��
=��Q�C�!H@�
=�zῡG��B�\C��H                                    Bx�:J�  T          @�Q�@�G���G����R�f�\C�4{@�G���\��=q�%C��                                    Bx�:Yd  �          @���@�  ��{��=q�t  C��f@�  �
=q����/\)C�H�                                    Bx�:h
  T          @��@�
=���У��{�
C���@�
=�������8(�C�g�                                    Bx�:v�  �          @���@�Q���H�����_33C�<)@�Q��{��  �(�C��                                    Bx�:�V  �          @�=q@�����  ��(���  C�B�@�������ff�F�\C���                                    Bx�:��  �          @��@��\�޸R��=q�q�C�T{@��\��\���1p�C���                                    Bx�:��  �          @�=q@����R����k33C��=@������
=�3�C�Ff                                    Bx�:�H  �          @���@����ff�����N�\C�]q@����ff�z�H���C�.                                    Bx�:��  �          @�=q@��׿Ǯ���
�z�C�q�@��׿޸R�+���=qC���                                    Bx�:Δ  �          @ƸR@�  ��Q�!G����C�AH@�  ������R�5�C��=                                    Bx�:�:  �          @�p�@����(���  ��C��@�����R=�\)?0��C��                                    Bx�:��  �          @��
@��R���þW
=�   C���@��R���=��
?=p�C��                                    Bx�:��  �          @�=q@���z�����r�\C�s3@���(��\)��ffC�,�                                    Bx�;	,  T          @\@�
=���׽�����C���@�
=����>�?�(�C���                                    Bx�;�  �          @���@�\)�:�H>��?�Q�C���@�\)�.{>��R@:�HC��                                     Bx�;&x  �          @\@�G��&ff=�?�z�C��=@�G��(�>��@{C�)                                    Bx�;5  �          @�  @�\)�u>���@s�
C���@�\)<�>���@u>��R                                    Bx�;C�  �          @�{@�G�>�z�?�  AB=q@:�H@�G�?\)?�33A333@���                                    Bx�;Rj  �          @��@�����H?�A�\)C�T{@�����@   A�=qC�L�                                    Bx�;a  �          @�  @�\)���@�A�=qC�Ff@�\)�W
=@G�A�C���                                    Bx�;o�  �          @���@�(���{@ffA�p�C��@�(��Q�@%A˙�C���                                    Bx�;~\  �          @�\)@�33�L��@!�A�=qC��H@�33�k�@(��A��C�˅                                    Bx�;�  �          @�@�G����
@<(�A�(�C�5�@�G���Q�@FffA�{C��3                                    Bx�;��  �          @�@�Q쿗
=@(�A���C���@�Q�(�@(��A��
C���                                    Bx�;�N  �          @�33@���2�\>\@p  C���@���(��?}p�A�
C�S3                                    Bx�;��  �          @�z�@����O\)��33�Y��C�C�@����O\)>�p�@g�C�G�                                    Bx�;ǚ  T          @�ff@���:�H<��
>#�
C�C�@���5?(��@��
C���                                    